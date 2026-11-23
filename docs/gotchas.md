# Coding gotchas

## Keep both ends of a copy until drain is proven

Await the receipt's drain outcome before releasing blocks. A lost reply can
leave a worker reading the source and writing the destination. On an
unproven outcome, keep the source blocks pinned and leave the destination
fence incomplete. Sending a load failure too early lets the scheduler reuse
blocks the worker can still overwrite.

Test by losing the reply while holding the running copy. Check that the
source stays pinned, the fence stays incomplete, and the scheduler gets no
load terminal.
[Lost-reply test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_drain.rs)

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
