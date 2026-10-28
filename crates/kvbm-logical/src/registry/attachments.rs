// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attachment system for storing arbitrary typed data on registration handles.

use super::handle::BlockRegistrationHandle;
use crate::events::protocol::EventReleaseHandle;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Error types for attachment operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentError {
    /// Attempted to attach a type as unique when it's already registered as multiple
    TypeAlreadyRegisteredAsMultiple(TypeId),
    /// Attempted to attach a type as multiple when it's already registered as unique
    TypeAlreadyRegisteredAsUnique(TypeId),
}

impl std::fmt::Display for AttachmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id) => {
                write!(
                    f,
                    "Type {:?} is already registered as multiple attachment",
                    type_id
                )
            }
            AttachmentError::TypeAlreadyRegisteredAsUnique(type_id) => {
                write!(
                    f,
                    "Type {:?} is already registered as unique attachment",
                    type_id
                )
            }
        }
    }
}

impl std::error::Error for AttachmentError {}

/// Tracks how a type is registered in the attachment system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttachmentMode {
    Unique,
    Multiple,
}

/// Storage for attachments on a BlockRegistrationHandle
#[derive(Debug)]
pub(crate) struct AttachmentStore {
    /// Unique attachments - only one value per TypeId
    pub(super) unique_attachments: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Multiple attachments - multiple values per TypeId
    pub(super) multiple_attachments: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
    /// Track which types are registered and how
    pub(super) type_registry: HashMap<TypeId, AttachmentMode>,
    /// Publisher of this registration's `Remove` event, present while an
    /// [`EventsManager`](crate::events::EventsManager) is attached to the
    /// registry. Both removal paths take it out under the entry's position
    /// guard, so the `Remove` cannot be reordered behind the `Create` of the
    /// registration that replaces this entry.
    pub(super) event_release: Option<EventReleaseHandle>,
    /// Refcounted physical-residency tracking for registered slots.
    /// Each `Primary`, `Duplicate`, `Inactive`, or `Held` slot for this hash
    /// and tier `T` contributes one count. `has_block::<T>` returns
    /// `count > 0`. The count does not prove request availability. Key is
    /// `TypeId::of::<T>()`.
    pub(crate) presence_markers: PresenceMarkers,
}

/// Tiers that [`PresenceMarkers`] keeps without a heap allocation.
const INLINE_PRESENCE_TIERS: usize = 2;

/// Presence counts for each metadata tier of one registration.
///
/// A registration is present in few tiers, so the counts live in a short
/// inline list. The first marker does not allocate, which a hash map does.
#[derive(Debug, Default)]
pub(crate) struct PresenceMarkers {
    inline: [Option<(TypeId, u32)>; INLINE_PRESENCE_TIERS],
    overflow: Vec<(TypeId, u32)>,
}

impl PresenceMarkers {
    fn count_mut(&mut self, type_id: TypeId) -> Option<&mut u32> {
        self.inline
            .iter_mut()
            .flatten()
            .chain(self.overflow.iter_mut())
            .find(|(id, _)| *id == type_id)
            .map(|(_, count)| count)
    }

    pub(crate) fn count(&self, type_id: TypeId) -> u32 {
        self.inline
            .iter()
            .flatten()
            .chain(self.overflow.iter())
            .find(|(id, _)| *id == type_id)
            .map_or(0, |(_, count)| *count)
    }

    pub(crate) fn increment(&mut self, type_id: TypeId) {
        if let Some(count) = self.count_mut(type_id) {
            *count += 1;
        } else if let Some(slot) = self.inline.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some((type_id, 1));
        } else {
            self.overflow.push((type_id, 1));
        }
    }

    /// Decrements the count of `type_id` and removes it at zero. Returns
    /// `false` if `type_id` has no count.
    pub(crate) fn decrement(&mut self, type_id: TypeId) -> bool {
        let Some(count) = self.count_mut(type_id) else {
            return false;
        };
        debug_assert!(*count > 0, "presence marker with a zero count");
        *count -= 1;
        if *count > 0 {
            return true;
        }
        if let Some(slot) = self
            .inline
            .iter_mut()
            .find(|slot| matches!(slot, Some((id, _)) if *id == type_id))
        {
            *slot = None;
        } else if let Some(index) = self.overflow.iter().position(|(id, _)| *id == type_id) {
            self.overflow.swap_remove(index);
        }
        true
    }
}

impl AttachmentStore {
    pub(super) fn new() -> Self {
        Self {
            unique_attachments: HashMap::new(),
            multiple_attachments: HashMap::new(),
            type_registry: HashMap::new(),
            event_release: None,
            presence_markers: PresenceMarkers::default(),
        }
    }
}

/// Typed accessor for attachments of a specific type
pub struct TypedAttachments<'a, T> {
    pub(super) handle: &'a BlockRegistrationHandle,
    pub(super) _phantom: PhantomData<T>,
}

impl<'a, T: Any + Send + Sync> TypedAttachments<'a, T> {
    /// Execute a closure with immutable access to the unique attachment of type T.
    pub fn with_unique<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();
        attachments
            .unique_attachments
            .get(&type_id)?
            .downcast_ref::<T>()
            .map(f)
    }

    /// Execute a closure with mutable access to the unique attachment of type T.
    pub fn with_unique_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();
        attachments
            .unique_attachments
            .get_mut(&type_id)?
            .downcast_mut::<T>()
            .map(f)
    }

    /// Execute a closure with immutable access to multiple attachments of type T.
    pub fn with_multiple<R>(&self, f: impl FnOnce(&[&T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();

        let multiple_refs: Vec<&T> = attachments
            .multiple_attachments
            .get(&type_id)
            .map(|vec| vec.iter().filter_map(|v| v.downcast_ref::<T>()).collect())
            .unwrap_or_default();

        f(&multiple_refs)
    }

    /// Execute a closure with mutable access to multiple attachments of type T.
    pub fn with_multiple_mut<R>(&self, f: impl FnOnce(&mut [&mut T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();

        let mut multiple_refs: Vec<&mut T> = attachments
            .multiple_attachments
            .get_mut(&type_id)
            .map(|vec| {
                vec.iter_mut()
                    .filter_map(|v| v.downcast_mut::<T>())
                    .collect()
            })
            .unwrap_or_default();

        f(&mut multiple_refs)
    }

    /// Execute a closure with immutable access to both unique and multiple attachments of type T.
    pub fn with_all<R>(&self, f: impl FnOnce(Option<&T>, &[&T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let attachments = self.handle.inner.attachments.lock();

        let unique = attachments
            .unique_attachments
            .get(&type_id)
            .and_then(|v| v.downcast_ref::<T>());

        let multiple_refs: Vec<&T> = attachments
            .multiple_attachments
            .get(&type_id)
            .map(|vec| vec.iter().filter_map(|v| v.downcast_ref::<T>()).collect())
            .unwrap_or_default();

        f(unique, &multiple_refs)
    }

    /// Execute a closure with mutable access to both unique and multiple attachments of type T.
    pub fn with_all_mut<R>(&self, f: impl FnOnce(Option<&mut T>, &mut [&mut T]) -> R) -> R {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.handle.inner.attachments.lock();

        match attachments.type_registry.get(&type_id) {
            Some(AttachmentMode::Unique) => {
                let unique = attachments
                    .unique_attachments
                    .get_mut(&type_id)
                    .and_then(|v| v.downcast_mut::<T>());
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(unique, &mut empty_vec)
            }
            Some(AttachmentMode::Multiple) => {
                let mut multiple_refs: Vec<&mut T> = attachments
                    .multiple_attachments
                    .get_mut(&type_id)
                    .map(|vec| {
                        vec.iter_mut()
                            .filter_map(|v| v.downcast_mut::<T>())
                            .collect()
                    })
                    .unwrap_or_default();
                f(None, &mut multiple_refs)
            }
            None => {
                let mut empty_vec: Vec<&mut T> = Vec::new();
                f(None, &mut empty_vec)
            }
        }
    }
}

#[cfg(test)]
mod presence_marker_tests {
    use super::*;

    struct TierA;
    struct TierB;
    struct TierC;

    #[test]
    fn counts_more_tiers_than_the_inline_list() {
        let ids = [
            TypeId::of::<TierA>(),
            TypeId::of::<TierB>(),
            TypeId::of::<TierC>(),
        ];
        let mut markers = PresenceMarkers::default();
        for (n, id) in ids.iter().enumerate() {
            for _ in 0..=n {
                markers.increment(*id);
            }
        }
        assert_eq!(ids.map(|id| markers.count(id)), [1, 2, 3]);
        assert!(markers.decrement(ids[0]));
        assert!(!markers.decrement(ids[0]));
        markers.increment(ids[2]);
        assert_eq!(ids.map(|id| markers.count(id)), [0, 2, 4]);
        for _ in 0..4 {
            assert!(markers.decrement(ids[2]));
        }
        assert_eq!(ids.map(|id| markers.count(id)), [0, 2, 0]);
        assert!(markers.overflow.is_empty() || markers.inline.iter().any(Option::is_none));
    }
}
