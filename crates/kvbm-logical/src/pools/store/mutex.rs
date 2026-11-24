// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::thread::{self, ThreadId};

#[cfg(not(test))]
use parking_lot::MutexGuard;
#[cfg(test)]
use tracing_mutex::parkinglot::MutexGuard;

use super::Mutex;

thread_local! {
    static ACQUISITIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) struct StoreMutex<T>(Mutex<T>);

impl<T> StoreMutex<T> {
    pub(super) fn new(value: T) -> Self {
        Self(Mutex::new(value))
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, T> {
        let guard = self.0.lock();
        let _ = ACQUISITIONS.try_with(|count| {
            if let Some(current) = count.get() {
                count.set(Some(current.saturating_add(1)));
            }
        });
        guard
    }
}

/// Run `f` and return its result and the number of block-store mutex acquisitions.
///
/// Enable `store-lock-counter` to use this function. Run the work and drop
/// its store handles inside `f` on the calling OS thread. Locks taken by
/// spawned work on other threads and drops after `f` returns fall outside
/// the count. Nested scopes count toward enclosing scopes, including on unwind.
pub fn count_store_lock_acquisitions<R>(f: impl FnOnce(&StoreLockCount) -> R) -> (R, usize) {
    ACQUISITIONS.with(|count| {
        let previous = count.get();
        let _reset = previous.is_none().then(|| {
            count.set(Some(0));
            ResetCount(count)
        });
        let scope = StoreLockCount {
            thread: thread::current().id(),
            start: previous.unwrap_or_default(),
        };
        let result = f(&scope);
        (result, scope.count())
    })
}

/// Access to the count while a measurement scope is active.
pub struct StoreLockCount {
    thread: ThreadId,
    start: usize,
}

impl StoreLockCount {
    /// Return the acquisitions on the originating thread since this scope began.
    ///
    /// # Panics
    ///
    /// Panics if called from a different OS thread.
    pub fn count(&self) -> usize {
        assert!(
            self.thread == thread::current().id(),
            "store lock count belongs to a different thread"
        );
        ACQUISITIONS.with(|count| count.get().expect("active store lock count") - self.start)
    }
}

struct ResetCount<'a>(&'a Cell<Option<usize>>);

impl Drop for ResetCount<'_> {
    fn drop(&mut self) {
        self.0.set(None);
    }
}
