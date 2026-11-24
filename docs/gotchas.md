# Coding gotchas

## Keep both ends of a copy until drain is proven

Await the receipt's drain outcome before releasing blocks. A lost reply can
leave a worker reading the source and writing the destination. On an
unproven outcome, keep the source blocks pinned and leave the destination
fence incomplete. Sending a load failure too early lets the scheduler reuse
blocks the worker can still overwrite.

Start an onboard over loopback and wait for the worker to start the copy.
Lose the transfer reply while the copy runs. With a fake clock, wait for the
client to see the loss before advancing the initial check timer. Check that
the source stays pinned, the fence stays incomplete, and the scheduler gets
no load terminal.
[Lost-reply test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_pause.rs)

## Report the undispatched tail of a failed onboard

If a serial resource or bundle onboard stops, report the failed copy and every
destination the driver never dispatched. vLLM invalidates only the named
blocks. A completion proof for a paused bundle copy preserves its landed
blocks and leaves the undispatched tail for recompute.

Test by failing the second of three resource copies. The first destination
stays valid and the failure names both the second and third.
[Bundle failure test](../crates/kvbm-engine/src/tiering/engine/bundle/integration_tests.rs)

## Resume the remaining resources after a lost reply

Proof for one resource copy covers only that copy. Keep the request pending
and route any remaining resources through admission before completing its
drain fence.

Test with two resources. Lose the first reply, prove the first copy drained,
and hold the second copy open. The sink must wait for the second proof.
[Resource restore test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_drain.rs)

## Stop CD decode pulls when a copy loses its reply

Keep the source pinned and the destination fenced until the worker proves
the copy drained. End the CD onboard at that proof and preserve the blocks
that landed. vLLM recomputes the rest, including any runs that were still
queued. Settle the originating CD session before releasing the fence.

Test with two disjoint remote runs in one availability batch. Lose the first
run's reply and prove it drained. The second run must never be pulled, and its
destination must be reported for recompute.
[CD decode drain tests](../crates/kvbm-engine/src/tiering/engine/local/tests/cd_tests/drain_proof.rs)

## Stash a failed prefill kick before settling it

After a prefill kick proves drain with an error, stash the failure, settle
the logical terminal, then settle the physical terminal before cleanup.
Connector release can race cleanup as soon as it sees a terminal. Without
the stash it can finalize the session and tell the decode peer it succeeded.

Test by releasing the prefill lifecycle as soon as the handle's logical
terminal is visible, while the physical fence is still pending. The release
must find the failure stash and close the session. The worker sink runs
after the physical terminal.
[Prefill logical-terminal test](../crates/kvbm-engine/src/tiering/engine/local/tests/cd_tests/prefill_drain.rs)

## Decide request release under the same lock as physical settlement

Request finish and the load driver share the request index lock when they
claim a release or publish drain proof. A complete fence can still have a
receive waiting for delivery. Track it through the worker sink call so
finish can join the receive or convert it to the request drain.

Test by pausing after the receive claim and again after fence publication.
Check both finished sets before delivery and verify one entry after it.
[Request release tests](../crates/kvbm-engine/src/tiering/engine/local/tests/request_release.rs)

When the engine says a finish is pending, keep the connector slot until the
worker's release reaches the scheduler. A logical terminal can precede
physical drain, and even a proven drain can have a worker reply in transit.
Test by presenting an empty scheduler output after proof, then delivering the
finished-set entry. [Connector finish tests](../crates/kvbm-connector/src/connector/leader/tests/request_finish.rs)

The scheduler can also consume a receive before the Velo sink returns.
Acknowledge that receive by action ID before dropping the onboard handle.
Test by holding the sink after it queues the receive, consuming the result,
then finishing with and without an in-flight save. Without the acknowledgment,
finish joins an entry the scheduler already consumed.
[Consumed receive test](../crates/kvbm-connector/src/connector/leader/tests/request_finish.rs)

A preempted request's old save can overlap its restored onboard even though
the worker waits on an eviction fence before the next forward pass. Defer
the restored match until the old drain proves completion. Test both delivery
orders and check that the scheduler accepts one terminal entry.
[Preemption release test](../crates/kvbm-connector/src/connector/leader/tests/request_finish/restore.rs)

## Keep paused onboard blocks through connector teardown

Keep the engine's pause registry alive while a worker can still touch its
blocks. Dropping the connector can race with a missing copy reply. If the
registry drops then, the pool can give its held blocks to another request.

Test with a fake clock. Pause an onboard, drop the connector owners, and check
that the hold and its metrics survive. Return proof from the original worker
and check that both the block and the engine are released.
[Teardown test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_pause.rs)

## Leave time for a probe to finish

A silent worker can use the full check deadline. The probe then sleeps
before it tries again. The watchdog must allow both waits and scheduling
delay before it reports a stalled probe.

Test with a check deadline longer than the probe interval. Stop the probe
and check that the watchdog waits out the full budget before it reports a
stall. [Probe budget test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_pause.rs)

## Drain CUDA work before releasing its buffers

Recording a completion event can fail while a CUDA copy is still in flight.
Keep the registrations if synchronizing the stream fails. A transform must
also keep its scratch table if an upload can still read it.

Test by failing completion event recording after launch. The dispatch must
return a receipt. Fail the stream synchronization to check that the receipt
reports an unproven drain and the registration stays live.
[Drain and registration test](../crates/kvbm-physical/src/transfer/executor/registration.rs)

## Keep registered storage when NIXL refuses deregistration

Explicitly deregister owned handles before releasing their storage. On error,
keep both until process exit and log the failure once. Dropping the handle
would retry deregistration. An explicit extraction panics after retaining the
region. Ordinary teardown logs the error and continues.

Test a counted failure after a real DRAM registration. Verify that the
handle and storage never drop, including when extraction unwinds.
[Deregistration tests](../crates/kvbm-memory/src/nixl.rs)

## Check the receipt before recording a layer event

A transfer can return a receipt after dispatch fails. Check the receipt
before recording a layer event or marking the copy successful. A caller
that owns the CUDA stream can continue immediately when its receipt is ready
and successful.

Test with a failed receipt on a caller-owned stream and a receipt that never
resolves. A timeout must keep the receipt's holds.
[Layer receipt tests](../crates/kvbm-engine/src/worker/layer_transfer.rs)

## Reserve completion before starting a staged transfer

Reserve the status event before posting a NIXL request. A late reservation
can fail after the backend starts accessing the bounce buffer.

Test by failing the status reservation on a staged read. Check that
dispatch returns the reservation error before NIXL posts a request.
[Staged status test](../crates/kvbm-physical/src/transfer/tests/planner_nixl.rs)

## Drop the mutex guard before an if let body can re-enter

An `if let` scrutinee can keep a temporary mutex guard alive through the
matching body. Copy or take the value in a separate statement before that
body calls code that can take the same lock or wait for another thread.

Test both orders of a reset and re-registration marker update. Pause each
operation after it releases the store mutex and verify that the other can
reach its own pause. [Presence ordering test](../crates/kvbm-logical/src/manager/tests/lineage_ghosts.rs)

## Keep scoped trace captures visible to parallel tests

Keep a second, disabled tracing dispatch alive during a scoped capture.
With a single dispatch, another test thread can register the same event
first and cache it as disabled because that thread lacks the capture.

Run the log assertion in the full parallel suite. It checks the worker
label even if another test logs a pause first.
[Pause worker test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_pause.rs)

## Check the prepared lineage under the lock that pins it

Check the manager, leaf tenure, and ordered physical source slots before
promoting any selected block. An ancestor can move to a new slot between a
separate check and pin while the leaf and every logical hash stay the same.

Test by preparing a root and child, releasing the root with reset, and
registering the root in another slot. Check that a hash-only pin accepts the
replacement and that the prepared pin rejects before changing any tenure.
[Relocation test](../crates/kvbm-logical/src/manager/tests/inactive_lineage_hold.rs)

## Scope a shared lineage hash by its resource

Keep one registry and manager for each logical resource, and look up a block
through that resource. A positional lineage hash names tokens, position, and
ancestry, not a loop pass or stream. Within one resource the store keeps one
primary per hash: `Reject` returns the existing primary and discards the new
payload, and `Allow` keeps a duplicate that a hash lookup cannot reach. The
untiered event stream carries no resource, so publish only the model's primary
resource.

Test two payloads with the same hash in one resource, then the same hash in two
resources with distinct registries. The first must return the existing primary;
the second must return each resource's own block.
[Resource identity tests](../crates/kvbm-logical/src/manager/tests/loop_resource.rs)
