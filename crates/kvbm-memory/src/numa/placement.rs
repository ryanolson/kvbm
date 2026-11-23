// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The NUMA placement decision for pinned host memory, and a reader for the
//! placement the kernel reports for a page.
//!
//! [`decide_placement`] is pure: it takes the device's node, the
//! NUMA-disabled flag and the CPU topology, and returns either a node to bind
//! or the reason the OS places the pages. [`page_placement`] asks the kernel
//! (`get_mempolicy`) what policy covers an address and which node holds its
//! page, so tests and callers can check the placement without trusting the
//! allocator's own record.

use nix::libc;
use serde::{Deserialize, Serialize};

use super::NumaNode;
use super::topology::{NodeMemorySnapshot, NumaTopology};
use crate::mmap_pinned::PlacementFallback;

/// The outcome of [`decide_placement`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlacementDecision {
    /// Bind the pages to this node, from a thread pinned to it.
    Bind(NumaNode),
    /// Let the OS place the pages, for this reason.
    OsDefault(PlacementFallback),
}

/// Decide where pinned host memory for a device goes.
///
/// Binds only to a known node that has CPUs. Checks, in order: NUMA disabled,
/// node unknown, topology unavailable, node without CPUs.
pub(crate) fn decide_placement(
    device_node: Option<NumaNode>,
    numa_disabled: bool,
    topology: Result<&NumaTopology, &str>,
) -> PlacementDecision {
    if numa_disabled {
        return PlacementDecision::OsDefault(PlacementFallback::NumaDisabled);
    }
    let node = match device_node {
        Some(node) if !node.is_unknown() => node,
        _ => return PlacementDecision::OsDefault(PlacementFallback::NodeUnknown),
    };
    let Ok(topology) = topology else {
        return PlacementDecision::OsDefault(PlacementFallback::TopologyUnavailable);
    };
    match topology.cpus_for_node(node.0) {
        Some(cpus) if !cpus.is_empty() => PlacementDecision::Bind(node),
        _ => PlacementDecision::OsDefault(PlacementFallback::NodeHasNoCpus(node)),
    }
}

/// Check whether a node's reclaimable memory estimate covers the mapped range.
pub(crate) fn check_node_memory_budget(
    node: NumaNode,
    mapped_bytes: u64,
    page_size: u64,
    read: impl FnOnce(NumaNode) -> Result<NodeMemorySnapshot, String>,
) -> Result<(), PlacementFallback> {
    let unavailable = |error| PlacementFallback::NodeMemoryBudgetUnavailable {
        node,
        mapped_bytes,
        error,
    };
    let snapshot = read(node).map_err(unavailable)?;
    let gross_bytes = [
        snapshot.inactive_file,
        snapshot.active_file,
        snapshot.reclaimable_slab,
    ]
    .into_iter()
    .try_fold(snapshot.free, u64::checked_add)
    .ok_or_else(|| unavailable("node memory counters overflow".into()))?
    .saturating_sub(snapshot.shmem)
    .min(snapshot.total);
    let reserve_bytes = snapshot
        .total
        .div_ceil(16)
        .checked_next_multiple_of(page_size)
        .ok_or_else(|| unavailable("node memory reserve cannot be rounded to a page".into()))?;
    let usable_bytes = gross_bytes.saturating_sub(reserve_bytes);
    tracing::debug!(
        ?node,
        mapped_bytes,
        usable_bytes,
        reserve_bytes,
        "checked node memory budget"
    );

    if mapped_bytes > usable_bytes {
        return Err(PlacementFallback::InsufficientNodeMemory {
            node,
            mapped_bytes,
            usable_bytes,
            reserve_bytes,
        });
    }
    Ok(())
}

/// A NUMA memory policy mode, as `get_mempolicy` reports it, without the
/// mode flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemPolicyMode {
    /// `MPOL_DEFAULT`: no policy on the range; the task's policy applies.
    Default,
    /// `MPOL_PREFERRED`.
    Preferred,
    /// `MPOL_BIND`.
    Bind,
    /// `MPOL_INTERLEAVE`.
    Interleave,
    /// `MPOL_LOCAL`.
    Local,
    /// `MPOL_PREFERRED_MANY`. On GB10 (driver 610.57.04),
    /// `cuMemHostRegister` turns an `MPOL_BIND` range into this mode with
    /// the same node mask.
    PreferredMany,
    /// `MPOL_WEIGHTED_INTERLEAVE`.
    WeightedInterleave,
    /// Any other mode number.
    Other(i32),
}

/// What the kernel reports for the page that holds one address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagePlacement {
    /// The policy mode on the range (`get_mempolicy(MPOL_F_ADDR)`).
    pub mode: MemPolicyMode,
    /// The nodes in the policy's node mask. Empty for [`MemPolicyMode::Default`].
    pub nodes: Vec<NumaNode>,
    /// The node that holds the page
    /// (`get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR)`). The kernel faults the
    /// page in if it is not yet present.
    pub node: NumaNode,
}

/// Ask the kernel for the memory policy on `addr` and the node of its page.
///
/// A diagnostic: it reads, and changes nothing except that the kernel faults
/// in a page that is not yet present. Fails with the `get_mempolicy` error,
/// for example `EPERM` under a seccomp profile that denies the call. The
/// policy is read first; if that fails, the node query (which faults the page
/// in) does not run.
pub fn page_placement(addr: usize) -> std::io::Result<PagePlacement> {
    let (mode, nodes) = read_policy(addr)?;
    let node = read_node(addr)?;
    Ok(PagePlacement { mode, nodes, node })
}

/// The two `get_mempolicy` answers for one page, each kept on its own, so a
/// failure of one query does not hide the other's answer.
#[derive(Debug)]
pub(crate) struct PageReading {
    /// The policy mode and node mask on the range (`MPOL_F_ADDR`).
    pub(crate) policy: std::io::Result<(MemPolicyMode, Vec<NumaNode>)>,
    /// The node that holds the page (`MPOL_F_NODE | MPOL_F_ADDR`).
    pub(crate) node: std::io::Result<NumaNode>,
}

impl From<std::io::Result<PagePlacement>> for PageReading {
    fn from(result: std::io::Result<PagePlacement>) -> Self {
        match result {
            Ok(p) => Self {
                policy: Ok((p.mode, p.nodes)),
                node: Ok(p.node),
            },
            Err(e) => Self {
                policy: Err(copy_error(&e)),
                node: Err(e),
            },
        }
    }
}

fn copy_error(e: &std::io::Error) -> std::io::Error {
    match e.raw_os_error() {
        Some(code) => std::io::Error::from_raw_os_error(code),
        None => std::io::Error::new(e.kind(), e.to_string()),
    }
}

/// Make both `get_mempolicy` queries for the page at `addr`, and keep each
/// answer.
pub(crate) fn read_page(addr: usize) -> PageReading {
    PageReading {
        policy: read_policy(addr),
        node: read_node(addr),
    }
}

fn read_policy(addr: usize) -> std::io::Result<(MemPolicyMode, Vec<NumaNode>)> {
    let mut mode: libc::c_int = 0;
    let mut mask = [0u64; MASK_BITS / 64];
    // SAFETY: `mode` and `mask` outlive the call; `MASK_BITS` is the bit
    // length of `mask`.
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut mode as *mut libc::c_int,
            mask.as_mut_ptr(),
            MASK_BITS as libc::c_ulong,
            addr,
            MPOL_F_ADDR,
        )
    };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let nodes = mask
        .iter()
        .enumerate()
        .flat_map(|(word, bits)| {
            (0..64u32)
                .filter(move |bit| bits & (1u64 << bit) != 0)
                .map(move |bit| NumaNode(word as u32 * 64 + bit))
        })
        .collect();
    Ok((MemPolicyMode::from_raw(mode & !MPOL_MODE_FLAGS), nodes))
}

fn read_node(addr: usize) -> std::io::Result<NumaNode> {
    let mut node: libc::c_int = 0;
    // SAFETY: `node` outlives the call; no mask is passed.
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut node as *mut libc::c_int,
            std::ptr::null_mut::<u64>(),
            0 as libc::c_ulong,
            addr,
            MPOL_F_NODE | MPOL_F_ADDR,
        )
    };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(NumaNode(node as u32))
}

/// `get_mempolicy` flags from `<numaif.h>`; `libc` does not export them.
const MPOL_F_NODE: libc::c_ulong = 1;
const MPOL_F_ADDR: libc::c_ulong = 2;
/// Policy modes from `<linux/mempolicy.h>` that `libc` does not export.
const MPOL_PREFERRED_MANY: libc::c_int = 5;
const MPOL_WEIGHTED_INTERLEAVE: libc::c_int = 6;
/// Mode flags the kernel may OR into the mode: `MPOL_F_NUMA_BALANCING`,
/// `MPOL_F_RELATIVE_NODES` and `MPOL_F_STATIC_NODES`.
const MPOL_MODE_FLAGS: libc::c_int = (1 << 13) | (1 << 14) | (1 << 15);
/// Bit length of the node mask passed to `get_mempolicy`. The kernel rejects
/// a mask shorter than its possible node count with `EINVAL`.
const MASK_BITS: usize = 1024;

impl MemPolicyMode {
    fn from_raw(mode: libc::c_int) -> Self {
        match mode {
            libc::MPOL_DEFAULT => Self::Default,
            libc::MPOL_PREFERRED => Self::Preferred,
            libc::MPOL_BIND => Self::Bind,
            libc::MPOL_INTERLEAVE => Self::Interleave,
            libc::MPOL_LOCAL => Self::Local,
            MPOL_PREFERRED_MANY => Self::PreferredMany,
            MPOL_WEIGHTED_INTERLEAVE => Self::WeightedInterleave,
            other => Self::Other(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Like gb300: node 0 has CPUs, nodes 1 and 2 are memory-only.
    fn topology() -> NumaTopology {
        NumaTopology::from_node_cpus(HashMap::from([
            (0, vec![0, 1, 2, 3]),
            (1, vec![]),
            (2, vec![]),
        ]))
    }

    #[test]
    fn placement_binds_a_node_with_cpus() {
        let topo = topology();
        assert_eq!(
            decide_placement(Some(NumaNode(0)), false, Ok(&topo)),
            PlacementDecision::Bind(NumaNode(0))
        );
    }

    #[test]
    fn placement_falls_back_when_the_node_is_unknown() {
        let topo = topology();
        assert_eq!(
            decide_placement(None, false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeUnknown)
        );
        assert_eq!(
            decide_placement(Some(NumaNode::UNKNOWN), false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeUnknown)
        );
    }

    #[test]
    fn placement_refuses_a_node_without_cpus() {
        let topo = topology();
        assert_eq!(
            decide_placement(Some(NumaNode(1)), false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeHasNoCpus(NumaNode(1)))
        );
        // A node the topology does not list has no CPUs either.
        assert_eq!(
            decide_placement(Some(NumaNode(7)), false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeHasNoCpus(NumaNode(7)))
        );
    }

    #[test]
    fn placement_falls_back_when_numa_is_disabled() {
        let topo = topology();
        assert_eq!(
            decide_placement(Some(NumaNode(0)), true, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NumaDisabled)
        );
    }

    #[test]
    fn placement_falls_back_without_a_topology() {
        assert_eq!(
            decide_placement(Some(NumaNode(0)), false, Err("no topology")),
            PlacementDecision::OsDefault(PlacementFallback::TopologyUnavailable)
        );
    }

    #[test]
    fn placement_fallback_serializes() {
        let f = PlacementFallback::BindRefused {
            node: NumaNode(3),
            error: "mbind: Invalid argument".into(),
        };
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<PlacementFallback>(&json).unwrap(), f);
    }
}
