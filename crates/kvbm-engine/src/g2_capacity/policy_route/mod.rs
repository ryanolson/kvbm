// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact G1-to-G2 route ownership for policy actions.

mod installation;
mod source;
mod state;
mod transaction;

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_logical::{BlockManager, InactiveLineageHold, ManagerId};
use kvbm_physical::transfer::{TransferDrainOutcome, TransferOptions};
use kvbm_protocols::connector::OffloadMode;
use tokio::sync::oneshot;

use super::{
    G2AllocationKind, G2Capacity, G2CapacityDecision, G2CapacityError, G2CapacityRequest,
    G2CapacityRequirement,
};
use crate::BlockId;
use crate::leader::InstanceLeader;
use installation::{ExactG1G2ManagerIdentity, ExactG1G2RouteIdentity, new_identity_pair};
use source::PolicyG1G2Source;
use transaction::{PolicyG1G2Transaction, PolicyG1G2TransactionOwner, TransactionRoute};

pub use installation::{
    PolicyG1G2Installation, PolicyG1G2ValidatedInstallation, PolicyG1SourceMetadata,
};

pub use state::{
    PolicyCancelDisposition, PolicyG1G2CancelHandle, PolicyG1G2Completion, PolicyG1G2Execution,
    PolicyG1G2ExecutionError, PolicyG1G2Reservation, PolicyG1G2SourceSettlement,
    PolicyG1G2SubmitError, PolicyPhysicalCompletion, PolicyPhysicalTerminal,
};

type TransferFuture = Pin<Box<dyn Future<Output = TransferDrainOutcome> + Send>>;

/// The private route half of one certified installation.
///
/// Only the unsafe constructor can mint an installation that contains this
/// value. Safe code cannot extract it from the installation.
/// This type has no reserve or submit operation.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::PolicyG1G2Route;
///
/// fn clone_unbound(route: &PolicyG1G2Route) -> PolicyG1G2Route {
///     route.clone()
/// }
/// ```
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::{PolicyG1G2Reservation, PolicyG1G2Route};
///
/// fn submit_raw(route: &PolicyG1G2Route, reservation: PolicyG1G2Reservation) {
///     route.submit(reservation, Vec::new());
/// }
/// ```
pub struct PolicyG1G2Route {
    core: Arc<PolicyG1G2RouteCore>,
    identity: ExactG1G2RouteIdentity,
}

/// One exact route bound to one logical source manager.
///
/// This capability is not cloneable. Each submission owns a real inactive
/// lineage hold.
#[doc(hidden)]
pub struct PolicyG1G2BoundRoute<T: PolicyG1SourceMetadata> {
    core: Arc<PolicyG1G2RouteCore>,
    binding: Arc<RouteBinding>,
    source_manager_id: ManagerId,
    _route_identity: ExactG1G2RouteIdentity,
    _manager_identity: ExactG1G2ManagerIdentity,
    source_type: PhantomData<fn() -> T>,
}

pub(super) struct PolicyG1G2RouteCore {
    resource: LogicalResourceId,
    capacity: Arc<dyn G2Capacity>,
    transfer: Arc<dyn PolicyG1G2TransferExecutor>,
    runtime: Option<tokio::runtime::Handle>,
}

pub(super) struct RouteBinding;

pub(crate) trait PolicyG1G2TransferExecutor: Send + Sync {
    fn execute(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_blocks: Vec<BlockId>,
        dst_blocks: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt>;
}

struct LeaderTransferExecutor {
    leader: InstanceLeader,
}

pub(crate) struct PolicyG1G2TransferReceipt {
    completion: TransferFuture,
}

impl PolicyG1G2Route {
    /// Certify one physical route and mint its complete installation.
    ///
    /// # Safety
    ///
    /// The leader and resource must identify the intended physical G1-to-G2
    /// transfer geometry. The caller must transfer the returned capability
    /// only to that resource's trusted logical factory. A violation can
    /// corrupt physical transfer state.
    ///
    /// ```compile_fail
    /// use kvbm_common::LogicalResourceId;
    /// use kvbm_engine::g2_capacity::PolicyG1G2Route;
    /// use kvbm_engine::leader::InstanceLeader;
    ///
    /// fn inject_safe(leader: InstanceLeader) {
    ///     let _ = PolicyG1G2Route::new(leader, LogicalResourceId(1));
    /// }
    /// ```
    #[allow(
        clippy::new_ret_no_self,
        reason = "the unsafe route constructor must return the indivisible installation"
    )]
    pub unsafe fn new(
        leader: InstanceLeader,
        resource: LogicalResourceId,
    ) -> Result<PolicyG1G2Installation, G2CapacityError> {
        let capacity = leader.g2_capacity_for(resource).ok_or_else(|| {
            G2CapacityError::Rejected(format!(
                "the leader has no G2 capacity for resource {resource:?}"
            ))
        })?;
        Ok(Self::installation_from_leader(leader, resource, capacity))
    }

    /// Certify one test route with a dedicated exact-capacity policy.
    ///
    /// This constructor does not replace the leader capacity. Compatibility
    /// routes continue to use the capacity configured on the leader.
    ///
    /// # Safety
    ///
    /// The leader and resource must identify the intended physical G1-to-G2
    /// transfer geometry. The caller must transfer the returned capability
    /// only to that resource's trusted logical factory. A violation can
    /// corrupt physical transfer state.
    #[cfg(test)]
    pub(crate) unsafe fn new_with_capacity(
        leader: InstanceLeader,
        resource: LogicalResourceId,
        capacity: Arc<dyn G2Capacity>,
    ) -> Result<PolicyG1G2Installation, G2CapacityError> {
        let manager = leader.g2_manager_for(resource).ok_or_else(|| {
            G2CapacityError::Rejected(format!(
                "the leader has no G2 manager for resource {resource:?}"
            ))
        })?;
        if capacity.manager_id() != manager.id() {
            return Err(G2CapacityError::Rejected(format!(
                "the dedicated G2 capacity does not own the leader G2 manager for resource {resource:?}"
            )));
        }

        Ok(Self::installation_from_leader(leader, resource, capacity))
    }

    fn installation_from_leader(
        leader: InstanceLeader,
        resource: LogicalResourceId,
        capacity: Arc<dyn G2Capacity>,
    ) -> PolicyG1G2Installation {
        let runtime = leader.runtime();
        let (route_identity, manager_identity) = new_identity_pair();
        let route = Self::from_parts(
            capacity,
            resource,
            Arc::new(LeaderTransferExecutor { leader }),
            Some(runtime),
            route_identity,
        );
        PolicyG1G2Installation {
            route,
            manager_identity,
        }
    }

    fn from_parts(
        capacity: Arc<dyn G2Capacity>,
        resource: LogicalResourceId,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        runtime: Option<tokio::runtime::Handle>,
        identity: ExactG1G2RouteIdentity,
    ) -> Self {
        Self {
            core: Arc::new(PolicyG1G2RouteCore {
                resource,
                capacity,
                transfer,
                runtime,
            }),
            identity,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_parts(
        capacity: Arc<dyn G2Capacity>,
        resource: LogicalResourceId,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
    ) -> PolicyG1G2Installation {
        let (route_identity, manager_identity) = new_identity_pair();
        let route = Self::from_parts(
            capacity,
            resource,
            transfer,
            tokio::runtime::Handle::try_current().ok(),
            route_identity,
        );
        PolicyG1G2Installation {
            route,
            manager_identity,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_parts_with_runtime(
        capacity: Arc<dyn G2Capacity>,
        resource: LogicalResourceId,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        runtime: tokio::runtime::Handle,
    ) -> PolicyG1G2Installation {
        let (route_identity, manager_identity) = new_identity_pair();
        let route = Self::from_parts(capacity, resource, transfer, Some(runtime), route_identity);
        PolicyG1G2Installation {
            route,
            manager_identity,
        }
    }
}

impl PolicyG1G2Installation {
    /// Return the certified physical resource.
    pub fn resource(&self) -> LogicalResourceId {
        self.route.core.resource
    }

    /// Validate and bind this capability to one typed G1 manager identity.
    pub fn validate<T: PolicyG1SourceMetadata>(
        self,
        source_resource: LogicalResourceId,
        source_manager: &BlockManager<T>,
    ) -> Result<PolicyG1G2ValidatedInstallation<T>, (G2CapacityError, Self)> {
        if self.resource() != source_resource {
            let error = G2CapacityError::Rejected(format!(
                "the exact route resource {:?} does not match source resource {source_resource:?}",
                self.resource()
            ));
            return Err((error, self));
        }
        debug_assert!(self.route.identity.matches_manager(&self.manager_identity));
        Ok(PolicyG1G2ValidatedInstallation {
            installation: self,
            manager_id: source_manager.id(),
            source_type: PhantomData,
        })
    }
}

impl<T: PolicyG1SourceMetadata> PolicyG1G2ValidatedInstallation<T> {
    /// Finish the already validated bind without a fallible operation.
    pub fn bind(self) -> PolicyG1G2BoundRoute<T> {
        let PolicyG1G2Installation {
            route,
            manager_identity,
        } = self.installation;
        PolicyG1G2BoundRoute {
            core: route.core,
            binding: Arc::new(RouteBinding),
            source_manager_id: self.manager_id,
            _route_identity: route.identity,
            _manager_identity: manager_identity,
            source_type: PhantomData,
        }
    }
}

impl<T: PolicyG1SourceMetadata> PolicyG1G2BoundRoute<T> {
    /// Recover the complete installation during a reversible factory rollback.
    #[doc(hidden)]
    pub fn into_installation(self) -> PolicyG1G2Installation {
        PolicyG1G2Installation {
            route: PolicyG1G2Route {
                core: self.core,
                identity: self._route_identity,
            },
            manager_identity: self._manager_identity,
        }
    }

    /// Reserve exact G2 capacity before source mutation.
    pub fn reserve(&self, count: usize) -> Result<PolicyG1G2Reservation, G2CapacityError> {
        if count == 0 {
            return Err(G2CapacityError::Rejected(
                "an exact G1-to-G2 transfer needs at least one block".to_string(),
            ));
        }
        let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::CacheExtension, count);
        let allocation = match self.core.capacity.reserve(request)? {
            G2CapacityDecision::ExactGranted(allocation) => allocation,
            G2CapacityDecision::Granted(_) => {
                return Err(G2CapacityError::RequirementMismatch {
                    expected: G2CapacityRequirement::ExactReclaim,
                    actual: G2CapacityRequirement::Compatibility,
                });
            }
            G2CapacityDecision::PendingReclaim(pending) => {
                return Err(G2CapacityError::PendingReclaim {
                    target_count: pending.plan().target_count(),
                });
            }
        };
        if allocation.kind() != G2AllocationKind::CacheExtension {
            return Err(G2CapacityError::Rejected(
                "the exact route received a different allocation kind".to_string(),
            ));
        }
        Ok(PolicyG1G2Reservation::new(
            allocation,
            Arc::clone(&self.binding),
        ))
    }

    /// Submit one owned logical source to an independent drain task.
    pub fn submit(
        &self,
        reservation: PolicyG1G2Reservation,
        source: InactiveLineageHold<T>,
        mode: OffloadMode,
    ) -> Result<PolicyG1G2Execution, PolicyG1G2SubmitError>
    where
        T: Send + 'static,
    {
        if source.manager_id() != self.source_manager_id {
            drop_source_before_terminal(source, reservation);
            return Err(PolicyG1G2SubmitError::ForeignSourceManager);
        }
        let Some(runtime) = self.core.runtime.clone() else {
            drop_source_before_terminal(source, reservation);
            return Err(PolicyG1G2SubmitError::NoRuntime);
        };
        let finished = Arc::new(AtomicBool::new(false));
        let (sender, completion) = oneshot::channel();
        let cancellation = reservation.cancellation();
        let owner = PolicyG1G2TransactionOwner::new(
            PolicyG1G2Transaction::new(
                TransactionRoute {
                    core: Arc::clone(&self.core),
                    binding: Arc::clone(&self.binding),
                },
                reservation,
                PolicyG1G2Source::new(source, mode),
            ),
            sender,
            Arc::clone(&finished),
        );
        let task_owner = owner.task_owner();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.spawn_blocking(move || task_owner.supervise())
        })) {
            Ok(_) => owner.commit_to_task(),
            Err(_) => {
                tracing::error!(
                    "Tokio rejected the exact G1-to-G2 blocking task before it started"
                );
                drop(owner);
            }
        }
        Ok(PolicyG1G2Execution::new(cancellation, completion, finished))
    }

    #[cfg(test)]
    pub(crate) fn abandon_before_task_start_for_test(
        &self,
        reservation: PolicyG1G2Reservation,
        source: InactiveLineageHold<T>,
        mode: OffloadMode,
    ) -> Result<PolicyG1G2Execution, PolicyG1G2SubmitError> {
        if source.manager_id() != self.source_manager_id {
            drop_source_before_terminal(source, reservation);
            return Err(PolicyG1G2SubmitError::ForeignSourceManager);
        }
        let finished = Arc::new(AtomicBool::new(false));
        let (sender, completion) = oneshot::channel();
        let cancellation = reservation.cancellation();
        let owner = PolicyG1G2TransactionOwner::new(
            PolicyG1G2Transaction::new(
                TransactionRoute {
                    core: Arc::clone(&self.core),
                    binding: Arc::clone(&self.binding),
                },
                reservation,
                PolicyG1G2Source::new(source, mode),
            ),
            sender,
            Arc::clone(&finished),
        );
        let task_owner = owner.task_owner();
        owner.commit_to_task();
        drop(task_owner);
        Ok(PolicyG1G2Execution::new(cancellation, completion, finished))
    }
}

fn drop_source_before_terminal<S, R>(source: S, reservation: R) {
    drop(source);
    drop(reservation);
}

impl PolicyG1G2TransferReceipt {
    pub(crate) fn new(completion: TransferFuture) -> Self {
        Self { completion }
    }

    fn drain(self) -> TransferDrainOutcome {
        futures::executor::block_on(self.completion)
    }
}

impl PolicyG1G2TransferExecutor for LeaderTransferExecutor {
    fn execute(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_blocks: Vec<BlockId>,
        dst_blocks: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt> {
        let notification = self.leader.execute_local_transfer_for_resource(
            resource, src, dst, src_blocks, dst_blocks, options,
        )?;
        Ok(PolicyG1G2TransferReceipt::new(Box::pin(async move {
            notification.await_drain().await
        })))
    }
}

#[cfg(test)]
mod ordering_tests {
    use std::sync::{Arc, Mutex};

    use super::drop_source_before_terminal;

    struct DropRecord {
        name: &'static str,
        order: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Drop for DropRecord {
        fn drop(&mut self) {
            self.order.lock().expect("drop order lock").push(self.name);
        }
    }

    #[test]
    fn synchronous_rejection_restores_source_before_terminal_owner_drop() {
        let order = Arc::new(Mutex::new(Vec::new()));

        drop_source_before_terminal(
            DropRecord {
                name: "source",
                order: Arc::clone(&order),
            },
            DropRecord {
                name: "reservation",
                order: Arc::clone(&order),
            },
        );

        assert_eq!(
            *order.lock().expect("drop order lock"),
            ["source", "reservation"]
        );
    }
}
