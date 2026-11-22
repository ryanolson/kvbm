// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Store growth: the slowest register, allocate and release chunk on a large
//! manager, for two workloads.
//!
//! This bench has its own `main` and no criterion. Criterion reports means,
//! and the defect this bench looks for is a rare, long pause: one chunk whose
//! call rebuilds a map under the store lock. So it times every call, keeps the
//! samples, and prints the chunk count, p50, p99 and max.
//!
//! Workloads:
//!
//! * `live` holds every guard. It registers chunks until every block is a live
//!   primary, then churns: each turnover drops the oldest quarter of the held
//!   guards (they go inactive), allocates a quarter (which evicts), stages it
//!   with fresh hashes, registers it and holds it. Live occupancy moves
//!   between 3C/4 and C. This exercises the store's map of live primaries.
//! * `inactive` registers a chunk and drops it, until the inactive pool is
//!   full, then churns the same way with eviction. At most one chunk is ever
//!   live. This exercises the inactive (lineage) index and its slab.
//!
//! A turnover is one quarter of the block count, in both workloads. A run
//! states its turnover count and claims nothing beyond it. The bench reports
//! time and occupancy only; it does not tell a map growth from an in-place
//! rehash.
//!
//! Each workload checks that it did what it claims, so a run that does not
//! hold or does not evict fails: `live` must peak at the block count of live
//! primaries, and `inactive` must count evictions equal to the blocks it
//! allocates after the fill.
//!
//! Usage (from `crates/`):
//!
//! ```text
//! cargo bench -p kvbm-logical --features testing --bench store_growth
//! cargo bench -p kvbm-logical --features testing --bench store_growth -- --quick
//! cargo bench -p kvbm-logical --features testing --bench store_growth -- \
//!     --blocks 262144 --turnovers 20 --workload live
//! ```
//!
//! `--bench --self-test` checks the bench's own argument parsing and
//! percentiles and exits. Without `--bench` in its arguments the binary
//! returns at once, whatever else it is given, so
//! `cargo test --all-targets` stays fast.

use std::collections::VecDeque;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use kvbm_logical::BlockManager;
use kvbm_logical::blocks::{BlockDuplicationPolicy, CompleteBlock, ImmutableBlock};
use kvbm_logical::events::EventsManager;
use kvbm_logical::registry::BlockRegistry;
use kvbm_logical::testing::TestMeta;
use kvbm_logical::tinylfu::TinyLFUTracker;

/// Tokens in each block.
const BLOCK_SIZE: u32 = 16;
/// Blocks in each timed call.
const CHUNK: usize = 1024;
/// Default block count: 2M blocks.
const DEFAULT_BLOCKS: usize = 2_097_152;
/// Default turnover count.
const DEFAULT_TURNOVERS: usize = 100;
/// Block count and turnover count of `--quick`.
const QUICK_BLOCKS: usize = 65_536;
const QUICK_TURNOVERS: usize = 4;

const USAGE: &str = "usage: store_growth --bench [--quick] [--blocks N] [--turnovers T] \
                     [--workload live|inactive|all] [--self-test]\n  \
                     --blocks N      block count; a multiple of 4096 (default 2097152)\n  \
                     --turnovers T   churn turnovers of C/4 blocks each (default 100)\n  \
                     --quick         65536 blocks, 4 turnovers: a smoke check, not a measurement\n  \
                     --self-test     check the bench's argument parsing and percentiles, then exit";

type Manager = BlockManager<TestMeta>;
type Guard = ImmutableBlock<TestMeta>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Live,
    Inactive,
    All,
}

struct Args {
    blocks: usize,
    turnovers: usize,
    which: Which,
}

/// Parse the arguments after the program name. Returns `Ok(None)` when
/// `--bench` is absent, without looking at any other argument:
/// `cargo test --all-targets` runs this binary with the test harness's own
/// flags (for example `--nocapture`), and that run must succeed and do nothing.
fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    if !args.iter().any(|arg| arg == "--bench") {
        return Ok(None);
    }
    let mut quick = false;
    let mut blocks = None;
    let mut turnovers = None;
    let mut which = Which::All;
    let mut args = args.iter().cloned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" | "--self-test" => {}
            "--quick" => quick = true,
            "--blocks" => blocks = Some(number(&arg, args.next())?),
            "--turnovers" => turnovers = Some(number(&arg, args.next())?),
            "--workload" => {
                which = match args.next().as_deref() {
                    Some("live") => Which::Live,
                    Some("inactive") => Which::Inactive,
                    Some("all") => Which::All,
                    other => {
                        return Err(format!("--workload takes live, inactive or all: {other:?}"));
                    }
                }
            }
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let blocks = blocks.unwrap_or(if quick { QUICK_BLOCKS } else { DEFAULT_BLOCKS });
    let turnovers = turnovers.unwrap_or(if quick {
        QUICK_TURNOVERS
    } else {
        DEFAULT_TURNOVERS
    });
    if blocks < 4 * CHUNK || blocks % (4 * CHUNK) != 0 {
        return Err(format!(
            "--blocks must be a multiple of {} and at least {}, got {blocks}",
            4 * CHUNK,
            4 * CHUNK
        ));
    }
    if turnovers == 0 {
        return Err("--turnovers must be at least 1".to_string());
    }
    Ok(Some(Args {
        blocks,
        turnovers,
        which,
    }))
}

fn number(flag: &str, value: Option<String>) -> Result<usize, String> {
    value
        .as_deref()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{flag} takes a number"))
}

/// A manager as in production: lineage backend, a registry with a TinyLFU
/// tracker and an events manager, `Reject` duplicates, 16-token blocks.
fn build_manager(blocks: usize) -> Manager {
    let registry = BlockRegistry::builder()
        .frequency_tracker(Arc::new(TinyLFUTracker::new(blocks)))
        .event_manager(Arc::new(EventsManager::builder().build()))
        .build();
    BlockManager::<TestMeta>::builder()
        .block_count(blocks)
        .block_size(BLOCK_SIZE as usize)
        .registry(registry)
        .duplication_policy(BlockDuplicationPolicy::Reject)
        .with_lineage_backend()
        .build()
        .expect("build manager")
}

/// One timed call's samples, in nanoseconds, in a preallocated buffer.
struct Samples(Vec<u64>);

impl Samples {
    fn with_capacity(calls: usize) -> Self {
        Self(Vec::with_capacity(calls))
    }

    fn push(&mut self, nanos: u128) {
        debug_assert!(self.0.len() < self.0.capacity(), "samples must not grow");
        // A u64 holds more than 500 years of nanoseconds, so a long pause is
        // never clipped to a shared value.
        self.0.push(u64::try_from(nanos).unwrap_or(u64::MAX));
    }

    /// Print one row: calls, p50, p99 and max in microseconds.
    fn print(&mut self, phase: &str, call: &str) {
        let n = self.0.len();
        let Some((p50, p99, max)) = percentiles(&mut self.0) else {
            println!(
                "  {phase:<6} {call:<9} {:>8} {:>10} {:>10} {:>10}",
                0, "-", "-", "-"
            );
            return;
        };
        let us = |nanos: u64| format!("{:.1}", nanos as f64 / 1e3);
        println!(
            "  {phase:<6} {call:<9} {n:>8} {:>10} {:>10} {:>10}",
            us(p50),
            us(p99),
            us(max)
        );
    }
}

/// Sort the samples, then return nearest-rank p50, p99 and the maximum, in the
/// samples' unit. `None` for no samples.
fn percentiles(samples: &mut [u64]) -> Option<(u64, u64, u64)> {
    let n = samples.len();
    if n == 0 {
        return None;
    }
    samples.sort_unstable();
    let at = |p: f64| samples[((p * n as f64).ceil() as usize).clamp(1, n) - 1];
    Some((at(0.50), at(0.99), samples[n - 1]))
}

/// Checks of the bench's own helpers, run by `--bench --self-test`. The
/// binary has no test harness, so `cargo test` does not run them.
fn self_test() {
    let words = |args: &[&str]| args.iter().map(|a| a.to_string()).collect::<Vec<_>>();
    // Test-harness flags without `--bench` must not be parsed.
    assert!(parse_args(&words(&[])).unwrap().is_none());
    assert!(
        parse_args(&words(&["--nocapture", "--test-threads=1"]))
            .unwrap()
            .is_none()
    );
    let quick = parse_args(&words(&["--bench", "--quick"]))
        .unwrap()
        .unwrap();
    assert_eq!(
        (quick.blocks, quick.turnovers),
        (QUICK_BLOCKS, QUICK_TURNOVERS)
    );
    assert!(parse_args(&words(&["--bench", "--blocks", "100"])).is_err());
    assert!(parse_args(&words(&["--bench", "--bogus"])).is_err());
    // Two long pauses, both over u32::MAX nanoseconds, must stay distinct.
    let mut samples = Samples::with_capacity(100);
    for nanos in (1..=98u128).map(|i| i * 1_000) {
        samples.push(nanos);
    }
    samples.push(5_000_000_000);
    samples.push(6_000_000_000);
    assert_eq!(
        percentiles(&mut samples.0),
        Some((50_000, 5_000_000_000, 6_000_000_000))
    );
    assert_eq!(percentiles(&mut []), None);
    println!("store_growth: self-test passed");
}

/// Stage `count` blocks with hashes no earlier chunk shares: one salt for
/// each chunk. Not timed.
fn stage(
    blocks: Vec<kvbm_logical::blocks::MutableBlock<TestMeta>>,
    salt: u64,
) -> Vec<CompleteBlock<TestMeta>> {
    let tokens: Vec<u32> = (0..(blocks.len() as u32 * BLOCK_SIZE)).collect();
    let sequence = dynamo_tokens::TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(salt));
    blocks
        .into_iter()
        .zip(sequence.blocks())
        .map(|(block, token_block)| block.complete(token_block).expect("stage"))
        .collect()
}

/// Time one `allocate_blocks_with_evictions` call. Returns the blocks and the
/// number evicted.
fn timed_allocate(
    manager: &Manager,
    samples: &mut Samples,
) -> (Vec<kvbm_logical::blocks::MutableBlock<TestMeta>>, usize) {
    let start = Instant::now();
    let (blocks, evicted) = manager
        .allocate_blocks_with_evictions(CHUNK)
        .expect("allocate a chunk");
    samples.push(start.elapsed().as_nanos());
    assert_eq!(blocks.len(), CHUNK);
    (blocks, evicted.len())
}

/// Time one `register_blocks` call.
fn timed_register(
    manager: &Manager,
    staged: Vec<CompleteBlock<TestMeta>>,
    samples: &mut Samples,
) -> Vec<Guard> {
    let count = staged.len();
    let start = Instant::now();
    let guards = manager.register_blocks(staged);
    samples.push(start.elapsed().as_nanos());
    assert_eq!(
        guards.len(),
        count,
        "every staged block must register as a live primary"
    );
    guards
}

/// Time dropping one chunk of guards. The vector keeps its capacity.
fn timed_release(guards: &mut Vec<Guard>, samples: &mut Samples) {
    let start = Instant::now();
    guards.clear();
    samples.push(start.elapsed().as_nanos());
}

/// The three call kinds of one phase.
struct Phase {
    allocate: Samples,
    register: Samples,
    release: Samples,
}

impl Phase {
    fn new(chunks: usize) -> Self {
        Self {
            allocate: Samples::with_capacity(chunks),
            register: Samples::with_capacity(chunks),
            release: Samples::with_capacity(chunks),
        }
    }

    fn print(&mut self, name: &str) {
        self.allocate.print(name, "allocate");
        self.register.print(name, "register");
        self.release.print(name, "release");
    }
}

/// Occupancy the run saw.
#[derive(Default)]
struct Peaks {
    live: usize,
    inactive: usize,
}

impl Peaks {
    fn observe(&mut self, manager: &Manager, live: usize) {
        self.live = self.live.max(live);
        self.inactive = self.inactive.max(manager.inactive_len());
    }
}

fn print_header(name: &str, blocks: usize, turnovers: usize) {
    println!(
        "\nworkload {name}: {blocks} blocks, {BLOCK_SIZE} tokens each, chunk {CHUNK}, \
         {turnovers} turnovers of {} blocks",
        blocks / 4
    );
    println!(
        "  {:<6} {:<9} {:>8} {:>10} {:>10} {:>10}",
        "phase", "call", "chunks", "p50 us", "p99 us", "max us"
    );
}

/// `live`: hold every guard to capacity, then churn a quarter at a time.
fn run_live(blocks: usize, turnovers: usize) {
    let manager = build_manager(blocks);
    let fill_chunks = blocks / CHUNK;
    let quarter_chunks = blocks / 4 / CHUNK;
    let churn_chunks = turnovers * quarter_chunks;
    let mut fill = Phase::new(fill_chunks);
    let mut churn = Phase::new(churn_chunks);
    let mut peaks = Peaks::default();
    // Preallocated, so the timed loop does not allocate. A scratch vector
    // takes each chunk of dropped guards.
    let mut held: VecDeque<Guard> = VecDeque::with_capacity(blocks + CHUNK);
    let mut dropping: Vec<Guard> = Vec::with_capacity(CHUNK);
    let evictions_before = manager.metrics().snapshot().evictions;
    let mut salt = 0u64;

    for _ in 0..fill_chunks {
        salt += 1;
        let (mutable, evicted) = timed_allocate(&manager, &mut fill.allocate);
        assert_eq!(evicted, 0, "the fill draws from the free pool");
        let staged = stage(mutable, salt);
        held.extend(timed_register(&manager, staged, &mut fill.register));
        peaks.observe(&manager, held.len());
    }
    assert_eq!(held.len(), blocks, "the fill must hold every block");
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.available_blocks(), 0);

    let mut evicted_total = 0usize;
    let mut allocated = 0usize;
    for _ in 0..turnovers {
        for _ in 0..quarter_chunks {
            dropping.extend(held.drain(..CHUNK));
            timed_release(&mut dropping, &mut churn.release);
            peaks.observe(&manager, held.len());
        }
        for _ in 0..quarter_chunks {
            salt += 1;
            let (mutable, evicted) = timed_allocate(&manager, &mut churn.allocate);
            evicted_total += evicted;
            allocated += CHUNK;
            let staged = stage(mutable, salt);
            held.extend(timed_register(&manager, staged, &mut churn.register));
            peaks.observe(&manager, held.len());
        }
        assert_eq!(held.len(), blocks, "each turnover returns to capacity");
    }

    // The checks that make a run that did not hold or did not evict fail.
    assert_eq!(
        peaks.live, blocks,
        "live: peak live primaries must equal the block count"
    );
    assert_eq!(
        evicted_total, allocated,
        "live: every churn allocation must evict from the inactive pool"
    );
    let counted = manager.metrics().snapshot().evictions - evictions_before;
    assert_eq!(counted as usize, allocated, "live: metrics disagree");

    print_header("live", blocks, turnovers);
    fill.print("fill");
    churn.print("churn");
    println!(
        "  peak live {} (block count {blocks}), peak inactive_len {}, turnovers {turnovers}, \
         evictions {evicted_total} of {allocated} post-fill allocations",
        peaks.live, peaks.inactive
    );
}

/// `inactive`: register and drop each chunk until the inactive pool fills,
/// then churn with eviction.
fn run_inactive(blocks: usize, turnovers: usize) {
    let manager = build_manager(blocks);
    let fill_chunks = blocks / CHUNK;
    let churn_chunks = turnovers * (blocks / 4 / CHUNK);
    let mut fill = Phase::new(fill_chunks);
    let mut churn = Phase::new(churn_chunks);
    let mut peaks = Peaks::default();
    let evictions_before = manager.metrics().snapshot().evictions;
    let mut salt = 0u64;

    for _ in 0..fill_chunks {
        salt += 1;
        let (mutable, evicted) = timed_allocate(&manager, &mut fill.allocate);
        assert_eq!(evicted, 0, "the fill draws from the free pool");
        let staged = stage(mutable, salt);
        let mut guards = timed_register(&manager, staged, &mut fill.register);
        peaks.observe(&manager, guards.len());
        timed_release(&mut guards, &mut fill.release);
        peaks.observe(&manager, 0);
    }
    assert_eq!(
        manager.inactive_len(),
        blocks,
        "the fill must fill the inactive pool"
    );
    assert_eq!(manager.available_blocks(), blocks);
    // The free pool is empty: every post-fill allocation must evict.
    assert_eq!(manager.reset_len(), 0, "the fill must use up the free pool");

    let mut evicted_total = 0usize;
    let mut allocated = 0usize;
    for _ in 0..churn_chunks {
        salt += 1;
        let (mutable, evicted) = timed_allocate(&manager, &mut churn.allocate);
        evicted_total += evicted;
        allocated += CHUNK;
        let staged = stage(mutable, salt);
        let mut guards = timed_register(&manager, staged, &mut churn.register);
        peaks.observe(&manager, guards.len());
        timed_release(&mut guards, &mut churn.release);
        peaks.observe(&manager, 0);
    }

    // The check that makes a run that never evicted fail.
    assert_eq!(
        evicted_total, allocated,
        "inactive: evictions must equal the blocks allocated after the fill"
    );
    let counted = manager.metrics().snapshot().evictions - evictions_before;
    assert_eq!(counted as usize, allocated, "inactive: metrics disagree");
    assert_eq!(manager.inactive_len(), blocks);

    print_header("inactive", blocks, turnovers);
    fill.print("fill");
    churn.print("churn");
    println!(
        "  peak live {} (one chunk at a time), peak inactive_len {}, turnovers {turnovers}, \
         evictions {evicted_total} of {allocated} post-fill allocations",
        peaks.live, peaks.inactive
    );
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(Some(args)) => args,
        // Not run as a bench: `cargo test --all-targets` runs this binary
        // with no arguments.
        Ok(None) => return ExitCode::SUCCESS,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("{message}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    if argv.iter().any(|arg| arg == "--self-test") {
        self_test();
        return ExitCode::SUCCESS;
    }
    println!(
        "store_growth: lineage backend, TinyLFU tracker, events manager, Reject duplicates; \
         times are per call over a {CHUNK}-block chunk"
    );
    if matches!(args.which, Which::Live | Which::All) {
        run_live(args.blocks, args.turnovers);
    }
    if matches!(args.which, Which::Inactive | Which::All) {
        run_inactive(args.blocks, args.turnovers);
    }
    ExitCode::SUCCESS
}
