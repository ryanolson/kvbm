// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;

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
/// The count covers all pools on the calling thread, including acquisitions
/// during cleanup. Nested scopes count toward their enclosing scopes, even
/// if a nested scope unwinds.
pub fn count_store_lock_acquisitions<R>(f: impl FnOnce() -> R) -> (R, usize) {
    ACQUISITIONS.with(|count| {
        let _scope = CountScope {
            count,
            previous: count.replace(Some(0)),
        };
        let result = f();
        (result, count.get().unwrap_or_default())
    })
}

struct CountScope<'a> {
    count: &'a Cell<Option<usize>>,
    previous: Option<usize>,
}

impl Drop for CountScope<'_> {
    fn drop(&mut self) {
        self.count.set(
            self.previous
                .map(|previous| previous.saturating_add(self.count.get().unwrap_or_default())),
        );
    }
}
