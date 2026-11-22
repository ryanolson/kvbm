//! Bounded module-cache refresh orchestration.

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use kvbm_protocols::control::{ControlError, LeaderControlClient, ModuleId};
use tokio::sync::Semaphore;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use velo_ext::InstanceId;

use velo_hub::store::{HubStore, Incarnation};

const MODULES_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
const MODULES_RPC_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_CONCURRENT_MODULE_REFRESHES: usize = 16;
const REGISTER_RETRY_DELAYS: &[Duration] = &[Duration::from_millis(500), Duration::from_secs(2)];

type TargetSnapshot = Arc<dyn Fn() -> Vec<ModuleTarget> + Send + Sync>;
type FetchModules = Arc<
    dyn Fn(InstanceId) -> BoxFuture<'static, Result<Vec<ModuleId>, ControlError>> + Send + Sync,
>;
type CommitModules = Arc<dyn Fn(ModuleTarget, Vec<ModuleId>) + Send + Sync>;

/// Owns all periodic and registration-triggered module refresh work.
pub(super) struct ModuleRefreshRuntime {
    inner: Arc<ModuleRefreshInner>,
    tasks: TaskTracker,
}

struct ModuleRefreshInner {
    cancel: CancellationToken,
    config: ModuleRefreshConfig,
    snapshot: TargetSnapshot,
    fetch: FetchModules,
    commit: CommitModules,
    permits: Arc<Semaphore>,
}

enum FetchOutcome {
    Committed,
    Failed(ControlError),
    TimedOut,
    Canceled,
}

#[derive(Clone, Copy)]
pub(super) struct ModuleTarget {
    instance_id: InstanceId,
    incarnation: Incarnation,
}

#[derive(Clone, Copy)]
struct ModuleRefreshConfig {
    periodic_interval: Duration,
    rpc_timeout: Duration,
    max_concurrent: usize,
    retry_delays: &'static [Duration],
}

impl ModuleTarget {
    pub(super) const fn new(instance_id: InstanceId, incarnation: Incarnation) -> Self {
        Self {
            instance_id,
            incarnation,
        }
    }

    pub(super) const fn instance_id(self) -> InstanceId {
        self.instance_id
    }

    pub(super) const fn incarnation(self) -> Incarnation {
        self.incarnation
    }
}

impl ModuleRefreshRuntime {
    pub(super) fn new<C>(
        cancel: CancellationToken,
        registry: Arc<dyn HubStore>,
        self_id: InstanceId,
        messenger: Arc<velo::Messenger>,
        commit: C,
    ) -> Self
    where
        C: Fn(ModuleTarget, Vec<ModuleId>) + Send + Sync + 'static,
    {
        let snapshot_registry = Arc::clone(&registry);
        Self::from_components(
            cancel,
            ModuleRefreshConfig::production(),
            move || {
                snapshot_registry
                    .list()
                    .into_iter()
                    .filter_map(|registered| {
                        let instance_id = registered.peer().instance_id();
                        (instance_id != self_id)
                            .then(|| ModuleTarget::new(instance_id, registered.incarnation))
                    })
                    .collect()
            },
            move |instance_id| {
                let client = LeaderControlClient::new(Arc::clone(&messenger), instance_id);
                Box::pin(async move { client.list_modules().await })
            },
            commit,
        )
    }

    #[cfg(test)]
    fn from_parts<F, C, S>(
        cancel: CancellationToken,
        config: ModuleRefreshConfig,
        snapshot: S,
        fetch: F,
        commit: C,
    ) -> Self
    where
        F: Fn(InstanceId) -> BoxFuture<'static, Result<Vec<ModuleId>, ControlError>>
            + Send
            + Sync
            + 'static,
        C: Fn(ModuleTarget, Vec<ModuleId>) + Send + Sync + 'static,
        S: Fn() -> Vec<ModuleTarget> + Send + Sync + 'static,
    {
        Self::from_components(cancel, config, snapshot, fetch, commit)
    }

    fn from_components<F, C, S>(
        cancel: CancellationToken,
        mut config: ModuleRefreshConfig,
        snapshot: S,
        fetch: F,
        commit: C,
    ) -> Self
    where
        F: Fn(InstanceId) -> BoxFuture<'static, Result<Vec<ModuleId>, ControlError>>
            + Send
            + Sync
            + 'static,
        C: Fn(ModuleTarget, Vec<ModuleId>) + Send + Sync + 'static,
        S: Fn() -> Vec<ModuleTarget> + Send + Sync + 'static,
    {
        config.max_concurrent = config.max_concurrent.max(1);
        Self {
            inner: Arc::new(ModuleRefreshInner {
                cancel: cancel.child_token(),
                config,
                snapshot: Arc::new(snapshot),
                fetch: Arc::new(fetch),
                commit: Arc::new(commit),
                permits: Arc::new(Semaphore::new(config.max_concurrent)),
            }),
            tasks: TaskTracker::new(),
        }
    }

    pub(super) fn spawn_periodic(&self) {
        let inner = Arc::clone(&self.inner);
        self.tasks.spawn(inner.run_periodic());
    }

    pub(super) fn spawn_on_register(&self, target: ModuleTarget) {
        if self.inner.cancel.is_cancelled() {
            return;
        }
        let inner = Arc::clone(&self.inner);
        self.tasks.spawn(inner.run_on_register(target));
    }

    #[cfg(test)]
    async fn shutdown(&self) {
        self.inner.cancel.cancel();
        self.tasks.close();
        self.tasks.wait().await;
    }
}

impl ModuleRefreshConfig {
    const fn production() -> Self {
        Self {
            periodic_interval: MODULES_REFRESH_INTERVAL,
            rpc_timeout: MODULES_RPC_TIMEOUT,
            max_concurrent: MAX_CONCURRENT_MODULE_REFRESHES,
            retry_delays: REGISTER_RETRY_DELAYS,
        }
    }
}

impl ModuleRefreshInner {
    async fn run_periodic(self: Arc<Self>) {
        let mut ticker = tokio::time::interval(self.config.periodic_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => return,
                _ = ticker.tick() => {}
            }
            let targets = (self.snapshot)();
            let refresh = self.refresh_targets(targets);
            tokio::select! {
                _ = self.cancel.cancelled() => return,
                _ = refresh => {}
            }
        }
    }

    async fn run_on_register(self: Arc<Self>, target: ModuleTarget) {
        let mut attempt = 0usize;
        loop {
            match self.fetch_and_commit(target).await {
                FetchOutcome::Committed | FetchOutcome::Canceled => return,
                FetchOutcome::Failed(error) => tracing::warn!(
                    instance = %target.instance_id,
                    attempt,
                    %error,
                    "control_plane: list_modules failed, retrying"
                ),
                FetchOutcome::TimedOut => tracing::warn!(
                    instance = %target.instance_id,
                    attempt,
                    timeout_ms = self.config.rpc_timeout.as_millis(),
                    "control_plane: list_modules timed out, retrying"
                ),
            }

            let Some(delay) = self.config.retry_delays.get(attempt).copied() else {
                tracing::warn!(
                    instance = %target.instance_id,
                    "control_plane: list_modules failed after backoff; cache stays empty until the periodic refresh"
                );
                return;
            };
            attempt += 1;
            tokio::select! {
                _ = self.cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
        }
    }

    async fn refresh_targets(self: &Arc<Self>, targets: Vec<ModuleTarget>) {
        stream::iter(targets)
            .for_each_concurrent(self.config.max_concurrent, |target| {
                let inner = Arc::clone(self);
                async move {
                    match inner.fetch_and_commit(target).await {
                        FetchOutcome::Failed(error) => tracing::debug!(
                            instance = %target.instance_id,
                            %error,
                            "control_plane: periodic list_modules failed"
                        ),
                        FetchOutcome::TimedOut => tracing::debug!(
                            instance = %target.instance_id,
                            timeout_ms = inner.config.rpc_timeout.as_millis(),
                            "control_plane: periodic list_modules timed out"
                        ),
                        FetchOutcome::Committed | FetchOutcome::Canceled => {}
                    }
                }
            })
            .await;
    }

    async fn fetch_and_commit(&self, target: ModuleTarget) -> FetchOutcome {
        let _permit = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => return FetchOutcome::Canceled,
            permit = Arc::clone(&self.permits).acquire_owned() => {
                permit.expect("module refresh semaphore is never closed")
            }
        };
        let fetch = (self.fetch)(target.instance_id);
        let result = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => return FetchOutcome::Canceled,
            result = tokio::time::timeout(self.config.rpc_timeout, fetch) => result,
        };
        match result {
            Ok(Ok(modules)) => {
                (self.commit)(target, modules);
                FetchOutcome::Committed
            }
            Ok(Err(error)) => FetchOutcome::Failed(error),
            Err(_) => FetchOutcome::TimedOut,
        }
    }
}

impl Drop for ModuleRefreshRuntime {
    fn drop(&mut self) {
        self.inner.cancel.cancel();
        self.tasks.close();
    }
}

#[cfg(test)]
mod tests {
    use std::future::pending;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use super::*;

    const TEST_INTERVAL: Duration = Duration::from_secs(60);
    const TEST_RPC_TIMEOUT: Duration = Duration::from_secs(5);
    const TEST_RETRY_DELAYS: &[Duration] = &[Duration::from_secs(30)];
    const LONG_RETRY_DELAYS: &[Duration] = &[Duration::from_secs(3_600)];

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn config() -> ModuleRefreshConfig {
        ModuleRefreshConfig {
            periodic_interval: TEST_INTERVAL,
            rpc_timeout: TEST_RPC_TIMEOUT,
            max_concurrent: 2,
            retry_delays: TEST_RETRY_DELAYS,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn healthy_periodic_peer_is_not_blocked_and_hung_rpc_times_out() {
        let hung = InstanceId::new_v4();
        let healthy = InstanceId::new_v4();
        let hung_dropped = Arc::new(AtomicUsize::new(0));
        let healthy_commits = Arc::new(AtomicUsize::new(0));
        let runtime = ModuleRefreshRuntime::from_parts(
            CancellationToken::new(),
            config(),
            move || {
                vec![
                    ModuleTarget::new(hung, Incarnation::from_u64(1)),
                    ModuleTarget::new(healthy, Incarnation::from_u64(1)),
                ]
            },
            {
                let hung_dropped = Arc::clone(&hung_dropped);
                move |instance_id| {
                    if instance_id == hung {
                        let hung_dropped = Arc::clone(&hung_dropped);
                        Box::pin(async move {
                            let _drop = DropCounter(hung_dropped);
                            pending::<Result<Vec<ModuleId>, ControlError>>().await
                        }) as BoxFuture<'static, _>
                    } else {
                        Box::pin(async { Ok(vec![ModuleId::Core]) }) as BoxFuture<'static, _>
                    }
                }
            },
            {
                let healthy_commits = Arc::clone(&healthy_commits);
                move |target, _| {
                    if target.instance_id == healthy {
                        healthy_commits.fetch_add(1, Ordering::SeqCst);
                    }
                }
            },
        );

        runtime.spawn_periodic();
        tokio::task::yield_now().await;
        tokio::time::advance(TEST_INTERVAL).await;
        tokio::task::yield_now().await;

        assert_eq!(
            healthy_commits.load(Ordering::SeqCst),
            1,
            "a hung peer head-of-line blocked the healthy periodic refresh"
        );
        assert_eq!(hung_dropped.load(Ordering::SeqCst), 0);

        tokio::time::advance(TEST_RPC_TIMEOUT).await;
        tokio::task::yield_now().await;
        assert_eq!(
            hung_dropped.load(Ordering::SeqCst),
            1,
            "the per-peer RPC did not time out"
        );
        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn cancellation_promptly_stops_periodic_and_on_register_fetches() {
        let target = ModuleTarget::new(InstanceId::new_v4(), Incarnation::from_u64(1));
        let started = Arc::new(tokio::sync::Semaphore::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let runtime = ModuleRefreshRuntime::from_parts(
            CancellationToken::new(),
            ModuleRefreshConfig {
                periodic_interval: Duration::from_millis(1),
                rpc_timeout: Duration::from_secs(3_600),
                ..config()
            },
            move || vec![target],
            {
                let started = Arc::clone(&started);
                let dropped = Arc::clone(&dropped);
                move |_| {
                    let started = Arc::clone(&started);
                    let dropped = Arc::clone(&dropped);
                    Box::pin(async move {
                        started.add_permits(1);
                        let _drop = DropCounter(dropped);
                        pending::<Result<Vec<ModuleId>, ControlError>>().await
                    }) as BoxFuture<'static, _>
                }
            },
            |_, _| {},
        );

        runtime.spawn_periodic();
        runtime.spawn_on_register(target);
        started
            .acquire_many(2)
            .await
            .expect("fetch start semaphore closed")
            .forget();

        tokio::time::timeout(Duration::from_millis(100), runtime.shutdown())
            .await
            .expect("module refresh shutdown hung behind an unbounded RPC");
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cancellation_interrupts_on_register_retry_backoff() {
        let target = ModuleTarget::new(InstanceId::new_v4(), Incarnation::from_u64(1));
        let calls = Arc::new(AtomicUsize::new(0));
        let first_call = Arc::new(tokio::sync::Semaphore::new(0));
        let runtime = ModuleRefreshRuntime::from_parts(
            CancellationToken::new(),
            ModuleRefreshConfig {
                retry_delays: LONG_RETRY_DELAYS,
                ..config()
            },
            Vec::new,
            {
                let calls = Arc::clone(&calls);
                let first_call = Arc::clone(&first_call);
                move |_| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    first_call.add_permits(1);
                    Box::pin(async { Err(ControlError::Internal("injected failure".to_owned())) })
                        as BoxFuture<'static, _>
                }
            },
            |_, _| {},
        );

        runtime.spawn_on_register(target);
        first_call
            .acquire()
            .await
            .expect("first-call semaphore closed")
            .forget();
        tokio::task::yield_now().await;

        tokio::time::timeout(Duration::from_millis(100), runtime.shutdown())
            .await
            .expect("module refresh shutdown hung in retry backoff");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn on_register_fetches_share_the_global_concurrency_bound() {
        let first = ModuleTarget::new(InstanceId::new_v4(), Incarnation::from_u64(1));
        let second = ModuleTarget::new(InstanceId::new_v4(), Incarnation::from_u64(1));
        let calls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let first_started = Arc::new(tokio::sync::Semaphore::new(0));
        let runtime = ModuleRefreshRuntime::from_parts(
            CancellationToken::new(),
            ModuleRefreshConfig {
                rpc_timeout: Duration::from_secs(3_600),
                max_concurrent: 1,
                ..config()
            },
            Vec::new,
            {
                let calls = Arc::clone(&calls);
                let dropped = Arc::clone(&dropped);
                let first_started = Arc::clone(&first_started);
                move |_| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    first_started.add_permits(1);
                    let dropped = Arc::clone(&dropped);
                    Box::pin(async move {
                        let _drop = DropCounter(dropped);
                        pending::<Result<Vec<ModuleId>, ControlError>>().await
                    }) as BoxFuture<'static, _>
                }
            },
            |_, _| {},
        );

        runtime.spawn_on_register(first);
        runtime.spawn_on_register(second);
        first_started
            .acquire()
            .await
            .expect("first-call semaphore closed")
            .forget();
        tokio::task::yield_now().await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "registration-triggered fetches exceeded the shared concurrency bound"
        );
        runtime.shutdown().await;
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn delayed_result_keeps_its_incarnation_for_commit_filtering() {
        let instance_id = InstanceId::new_v4();
        let stale = Incarnation::from_u64(1);
        let current = Arc::new(AtomicU64::new(stale.as_u64()));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let commits = Arc::new(AtomicUsize::new(0));
        let runtime = ModuleRefreshRuntime::from_parts(
            CancellationToken::new(),
            config(),
            Vec::new,
            {
                let release = Arc::clone(&release);
                move |_| {
                    let release = Arc::clone(&release);
                    Box::pin(async move {
                        release
                            .acquire()
                            .await
                            .expect("release semaphore closed")
                            .forget();
                        Ok(vec![ModuleId::Core])
                    }) as BoxFuture<'static, _>
                }
            },
            {
                let current = Arc::clone(&current);
                let commits = Arc::clone(&commits);
                move |target, _| {
                    if target.incarnation.as_u64() == current.load(Ordering::SeqCst) {
                        commits.fetch_add(1, Ordering::SeqCst);
                    }
                }
            },
        );

        runtime.spawn_on_register(ModuleTarget::new(instance_id, stale));
        tokio::task::yield_now().await;
        current.store(2, Ordering::SeqCst);
        release.add_permits(1);
        tokio::task::yield_now().await;

        assert_eq!(commits.load(Ordering::SeqCst), 0);
        runtime.shutdown().await;
    }
}
