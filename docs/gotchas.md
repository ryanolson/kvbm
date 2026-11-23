# Coding gotchas

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

After a CUDA copy reaches a stream, an error recording its completion event
does not prove that the copy drained. Keep the registrations if synchronizing
the stream fails. A transform must also keep its scratch table if an upload
can still read it.

Test by failing completion event recording after launch. The dispatch must
return a receipt. Fail the stream synchronization to check that the receipt
reports an unproven drain and the registration stays live.
[Drain and registration test](../crates/kvbm-physical/src/transfer/executor/registration.rs)
