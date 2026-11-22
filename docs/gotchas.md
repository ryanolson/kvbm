# Coding gotchas

## Keep paused onboard blocks through connector teardown

Keep the engine's pause registry alive while a worker can still touch its
blocks. Dropping the connector can race with a missing copy reply. If the
registry drops then, the pool can give its held blocks to another request.

Test with a fake clock. Pause an onboard, drop the connector owners, and check
that the hold and its metrics survive. Return proof from the original worker
and check that both the block and the engine are released.
[Teardown test](../crates/kvbm-engine/src/tiering/engine/local/tests/onboard_pause.rs)
