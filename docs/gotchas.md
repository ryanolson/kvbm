# Coding gotchas

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
