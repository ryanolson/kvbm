// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EventsManager for coordinating block registration events.
//!
//! The EventsManager hooks into BlockRegistry to emit KvCacheEvents when blocks
//! are registered or removed. It uses a policy to filter which blocks trigger events
//! and a broadcast channel to allow multiple subscribers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use derive_builder::Builder;
use futures::Stream;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use super::policy::EventEmissionPolicy;
use super::protocol::{EventReleaseHandle, KvCacheEvent};
use crate::registry::BlockRegistrationHandle;

/// Settings for constructing an [`EventsManager`].
///
/// # Example
///
/// ```ignore
/// // Simple with defaults (AllEventsPolicy)
/// let manager = EventsManagerSettings::builder().build()?.into_manager();
///
/// // With custom policy
/// let manager = EventsManagerSettings::builder()
///     .policy(Arc::new(PowerOfTwoPolicy::new()))
///     .build()?
///     .into_manager();
///
/// // With custom configuration
/// let manager = EventsManagerSettings::builder()
///     .channel_capacity(2048)
///     .build()?
///     .into_manager();
/// ```
#[derive(Builder, Clone)]
#[builder(setter(into, strip_option), build_fn(error = "anyhow::Error"))]
pub struct EventsManagerSettings {
    /// The event emission policy.
    ///
    /// Default: [`AllEventsPolicy`](super::policy::AllEventsPolicy)
    #[builder(default, setter(strip_option = false))]
    policy: Option<Arc<dyn EventEmissionPolicy>>,

    /// Capacity of the broadcast channel.
    ///
    /// Default: 1024
    #[builder(default = "1024")]
    channel_capacity: usize,
}

impl EventsManagerSettings {
    /// Creates a new builder for EventsManagerSettings.
    pub fn builder() -> EventsManagerSettingsBuilder {
        EventsManagerSettingsBuilder::default()
    }

    /// Converts settings into an EventsManager.
    pub fn into_manager(self) -> EventsManager {
        let policy = self
            .policy
            .unwrap_or_else(|| Arc::new(super::policy::AllEventsPolicy::new()));
        let (event_tx, _) = broadcast::channel(self.channel_capacity);

        EventsManager {
            policy,
            event_tx,
            lagged_events: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Manager for emitting and coordinating block registration events.
///
/// The EventsManager is responsible for:
/// - Filtering block registrations based on a policy
/// - Emitting Create events when blocks are registered
/// - Giving each registration the RAII handle that emits its Remove event
/// - Broadcasting events to multiple subscribers via [`subscribe()`](Self::subscribe)
///
/// Note: Instance context is applied at the publisher level via
/// [`KvbmCacheEventsPublisher`](super::publisher::KvbmCacheEventsPublisher).
///
/// # Example
///
/// ```ignore
/// // Create with defaults (AllEventsPolicy)
/// let manager = EventsManager::builder().build();
///
/// // Create with PowerOfTwoPolicy
/// let manager = EventsManager::builder()
///     .policy(Arc::new(PowerOfTwoPolicy::new()))
///     .build();
/// ```
pub struct EventsManager {
    policy: Arc<dyn EventEmissionPolicy>,
    event_tx: broadcast::Sender<KvCacheEvent>,
    /// Count of events a slow subscriber never saw, summed over the
    /// `Lagged(n)` reports of every subscriber stream. A dropped `Remove`
    /// is request-visible. `query_holders` returns the deepest hash with a
    /// holder, so one stale entry hides a live shallower holder until the
    /// hub reaps the instance. This counter is the signal that loss
    /// happened. It names no event and repairs nothing on its own.
    lagged_events: Arc<AtomicU64>,
}

/// Builder for [`EventsManager`] that wraps [`EventsManagerSettingsBuilder`].
#[derive(Default)]
pub struct EventsManagerBuilder(EventsManagerSettingsBuilder);

impl EventsManagerBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the event emission policy.
    ///
    /// Default: [`AllEventsPolicy`](super::policy::AllEventsPolicy)
    pub fn policy(mut self, policy: Arc<dyn EventEmissionPolicy>) -> Self {
        self.0.policy = Some(Some(policy));
        self
    }

    /// Sets the broadcast channel capacity.
    ///
    /// Default: 1024
    pub fn channel_capacity(mut self, capacity: usize) -> Self {
        self.0.channel_capacity = Some(capacity);
        self
    }

    /// Builds the EventsManager.
    pub fn build(self) -> EventsManager {
        self.0
            .build()
            .expect("EventsManagerSettings has all defaults")
            .into_manager()
    }
}

impl EventsManager {
    /// Creates a new builder for EventsManager.
    pub fn builder() -> EventsManagerBuilder {
        EventsManagerBuilder::new()
    }

    /// Subscribe to the event stream.
    ///
    /// Returns a stream of events. Multiple subscribers are supported, and each
    /// subscriber receives all events. Late subscribers will miss events that
    /// occurred before subscribing.
    ///
    /// A slow subscriber makes the broadcast channel report `Lagged(n)` for
    /// the `n` events it overwrote. The stream adds `n` to
    /// [`Self::lagged_events`], logs a warning, and continues with the next
    /// event. A dropped `Remove` hides a live holder from a later query, so
    /// the stream counts the loss even though it does not repair it.
    pub fn subscribe(&self) -> impl Stream<Item = KvCacheEvent> + Send + 'static {
        let rx = self.event_tx.subscribe();
        let lagged_events = Arc::clone(&self.lagged_events);
        BroadcastStream::new(rx).filter_map(move |result| match result {
            Ok(event) => Some(event),
            Err(BroadcastStreamRecvError::Lagged(n)) => {
                lagged_events.fetch_add(n, Ordering::Relaxed);
                tracing::warn!(dropped = n, "events subscriber lagged and lost events");
                None
            }
        })
    }

    /// Cumulative count of events dropped, summed over the `Lagged(n)`
    /// reports of every subscriber stream since this manager was built.
    pub fn lagged_events(&self) -> u64 {
        self.lagged_events.load(Ordering::Relaxed)
    }

    /// Hook called when a block is registered in the BlockRegistry.
    ///
    /// This method:
    /// 1. Checks the policy to determine if an event should be emitted
    /// 2. Broadcasts a Create event if the policy allows
    /// 3. Gives the registration handle the RAII publisher of its Remove event
    ///
    /// `register_sequence_hash` calls this under the entry's position guard, so
    /// both steps happen inside the critical section that orders this `Create`
    /// against the `Remove` of the registration it replaces.
    ///
    /// # Arguments
    /// * `handle` - The block registration handle
    pub fn on_block_registered(&self, handle: &BlockRegistrationHandle) {
        let seq_hash = handle.seq_hash();

        // Check policy - only emit events for filtered blocks
        if !self.policy.should_emit(seq_hash) {
            return;
        }

        // Emit Create event
        let create_event = KvCacheEvent::Create(seq_hash);

        // Broadcast send only fails if there are no receivers, which is fine
        let _ = self.event_tx.send(create_event);

        handle.attach_event_release(EventReleaseHandle::new(seq_hash, self.event_tx.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::super::policy::PowerOfTwoPolicy;
    use super::*;
    use crate::registry::BlockRegistry;
    use crate::{KvbmSequenceHashProvider, SequenceHash};
    use dynamo_tokens::TokenBlockSequence;
    use futures::StreamExt;

    fn create_seq_hash_at_position(position: usize) -> SequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].kvbm_sequence_hash()
    }

    #[tokio::test]
    async fn test_events_manager_emits_create_for_power_of_two() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(16); // Power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        // Register the block
        manager.on_block_registered(&handle);

        // Should receive Create event
        let event = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event, KvCacheEvent::Create(seq_hash));
    }

    #[tokio::test]
    async fn test_events_manager_skips_non_power_of_two() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(17); // Not power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        // Register the block
        manager.on_block_registered(&handle);

        // Should NOT receive any event (will timeout)
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(50), stream.next()).await;
        assert!(result.is_err()); // Timeout expected

        // Keep handle alive to prevent drop event
        drop(handle);
    }

    #[tokio::test]
    async fn test_events_manager_emits_remove_on_drop() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(32); // Power of 2

        {
            let handle = registry.register_sequence_hash(seq_hash);
            manager.on_block_registered(&handle);

            // Consume Create event
            let event = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(event, KvCacheEvent::Create(seq_hash));

            // Handle is dropped here, triggering Remove
        }

        // Should receive Remove event
        let event = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event, KvCacheEvent::Remove(seq_hash));
    }

    #[tokio::test]
    async fn test_events_manager_multiple_subscribers() {
        let manager = EventsManager::builder()
            .policy(Arc::new(PowerOfTwoPolicy::new()))
            .build();

        let mut stream1 = Box::pin(manager.subscribe());
        let mut stream2 = Box::pin(manager.subscribe());

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(64); // Power of 2
        let handle = registry.register_sequence_hash(seq_hash);

        manager.on_block_registered(&handle);

        // Both streams should receive the same event
        let event1 = tokio::time::timeout(std::time::Duration::from_millis(100), stream1.next())
            .await
            .unwrap()
            .unwrap();
        let event2 = tokio::time::timeout(std::time::Duration::from_millis(100), stream2.next())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(event1, KvCacheEvent::Create(seq_hash));
        assert_eq!(event2, KvCacheEvent::Create(seq_hash));
    }

    #[test]
    fn test_events_manager_default_policy() {
        // With no policy specified, should use AllEventsPolicy (default)
        let manager = EventsManager::builder().build();

        let registry = BlockRegistry::new();
        let seq_hash = create_seq_hash_at_position(17); // Not power of 2

        // Use a subscriber to verify events are emitted
        let _subscription = manager.subscribe();

        let handle = registry.register_sequence_hash(seq_hash);

        // With AllEventsPolicy, all blocks should emit events
        // (this would fail with PowerOfTwoPolicy for position 17)
        manager.on_block_registered(&handle);
    }

    #[tokio::test]
    async fn lagged_events_are_counted() {
        let manager = Arc::new(EventsManager::builder().channel_capacity(2).build());
        let mut stream = Box::pin(manager.subscribe());

        let registry = BlockRegistry::builder()
            .event_manager(Arc::clone(&manager))
            .build();

        // Five registrations into a two-slot channel, with the stream never
        // polled in between. The broadcast sender overwrites the three
        // oldest before any subscriber reads them. Keep every handle alive
        // so a Remove never competes with the Create events under test.
        let mut handles = Vec::with_capacity(5);
        for position in 0..5 {
            let seq_hash = create_seq_hash_at_position(position);
            handles.push(registry.register_sequence_hash(seq_hash));
        }

        let mut received = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(std::time::Duration::from_millis(100), stream.next()).await
        {
            received.push(event);
        }

        assert_eq!(
            received.len(),
            2,
            "only the two surviving Create events arrive"
        );
        assert_eq!(
            manager.lagged_events(),
            3,
            "the three overwritten events must be counted, not silently dropped"
        );
    }
}
