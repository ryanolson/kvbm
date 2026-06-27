// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[allow(unused_imports)]
use pyo3::exceptions::PyException;
#[allow(unused_imports)]
use pyo3::prelude::*;
use std::fmt::Display;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

/// Opt-out for hosts that install their own subscriber (mirrors Dynamo's
/// `DYNAMO_SKIP_PYTHON_LOG_INIT`, without depending on the Dynamo runtime).
const SKIP_LOG_INIT_ENV: &str = "KVBM_SKIP_LOG_INIT";

static LOG_INIT: Once = Once::new();

/// Install a process-wide tracing subscriber once, when this extension module
/// is imported.
///
/// KVBM runs as a library inside a host process — for the connector path,
/// vLLM's EngineCore subprocess — that installs no tracing subscriber, so the
/// connector's `kvbm_audit` / `kvbm_connector` events are dropped no matter
/// what `RUST_LOG` is set to. (Under the old v1 connector this came for free
/// because importing the `dynamo` runtime initialized logging; the v1 removal
/// dropped it.) This mirrors that import-time init pattern but stays standalone
/// — KVBM does not depend on the Dynamo runtime. The filter is read from
/// `RUST_LOG`, falling back to `DYN_LOG`, then an `error` default. `Once` +
/// `try_init` make it a safe no-op if a subscriber already exists.
fn init_logging() {
    if std::env::var_os(SKIP_LOG_INIT_ENV).is_some() {
        return;
    }
    LOG_INIT.call_once(|| {
        // Default ERROR (not warn): the CD execution path emits per-request
        // tracing::info! (decode_gnmt etc.) plus the kvbm_audit flood at info, so a
        // higher default floods long runs. Durable observability lives in the
        // kvbm-observability Prometheus metrics (independent of the log level). An
        // explicit RUST_LOG/DYN_LOG always overrides (e.g. the harness sets
        // warn,kvbm_audit=info; raise to info/debug to debug a bringup).
        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_from_env("DYN_LOG"))
            .unwrap_or_else(|_| EnvFilter::new("error"));
        // Strip ANSI escapes by default: the connector logs from a host
        // subprocess whose stdout is redirected to a file, where color codes
        // are just noise that breaks naive log greps.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .try_init();
    });
}

// TODO: kernels module needs adaptation - operational_copy API doesn't exist in decomposed kvbm-kernels
// #[cfg(feature = "kernels")]
// mod kernels;

mod connector;
mod runtime;
mod torch;
mod vllm;

#[cfg(feature = "hub")]
mod hub;

/// Check if the core connector bindings are available.
#[pyfunction]
fn is_available() -> bool {
    true
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    init_logging();

    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_class::<vllm::PyKvbmVllmConfig>()?;
    m.add_class::<torch::PyTensor>()?;
    m.add_class::<runtime::PyKvbmRuntime>()?;
    m.add_class::<connector::leader::PyConnectorLeader>()?;
    m.add_class::<connector::worker::PyConnectorWorker>()?;
    m.add_class::<connector::leader::PyRequest>()?;
    m.add_class::<connector::leader::PySchedulerOutput>()?;

    #[cfg(feature = "hub")]
    {
        let hub = PyModule::new(m.py(), "hub")?;
        hub::add_to_module(&hub)?;
        m.add_submodule(&hub)?;
    }

    // TODO: kernels bindings disabled pending operational_copy API adaptation
    // #[cfg(feature = "kernels")]
    // {
    //     let kernels = PyModule::new(m.py(), "kernels")?;
    //     kernels::add_to_module(&kernels)?;
    //     m.add_submodule(&kernels)?;
    // }

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}
