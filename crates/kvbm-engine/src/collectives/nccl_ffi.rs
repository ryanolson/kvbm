// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal NCCL ABI with versioned-library loading.
//!
//! Python CUDA wheels commonly ship `libnccl.so.2` without the unversioned
//! development symlink. Loading the small API surface used by KVBM here keeps
//! runtime collectives independent of that symlink.

use std::collections::HashSet;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Result, anyhow};
use libloading::Library;

pub(super) type NcclComm = *mut c_void;
pub(super) type CudaStream = *mut c_void;
pub(super) type NcclResult = c_int;

pub(super) const NCCL_SUCCESS: NcclResult = 0;
pub(super) const NCCL_IN_PROGRESS: NcclResult = 7;
pub(super) const NCCL_INT8: c_int = 0;
const NCCL_CONFIG_MAGIC: u32 = 0xcafebeef;
const NCCL_CONFIG_ABI_VERSION: u32 = 21400;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct NcclUniqueId {
    pub(super) internal: [c_char; 128],
}

/// The exact `ncclConfig_v21400` ABI introduced in NCCL 2.14.
///
/// NCCL uses `version` to decide which config fields the caller supplied. It
/// does not infer that solely from `size`, so this must advertise the 2.14 ABI
/// rather than the version of the dynamically loaded NCCL runtime. The final
/// member initializes the four bytes that were tail padding in the C struct;
/// those bytes overlap `cgaClusterSize` in newer NCCL layouts.
#[repr(C)]
pub(super) struct NcclConfig {
    size: usize,
    magic: u32,
    version: u32,
    blocking: c_int,
    legacy_tail_padding: c_int,
}

impl NcclConfig {
    pub(super) fn nonblocking(runtime_version: c_int) -> Result<Self> {
        let runtime_version = u32::try_from(runtime_version)
            .map_err(|_| anyhow!("NCCL reported invalid version code {runtime_version}"))?;
        if runtime_version < NCCL_CONFIG_ABI_VERSION {
            return Err(anyhow!(
                "NCCL {runtime_version} is too old for nonblocking communicator initialization; \
                 NCCL {NCCL_CONFIG_ABI_VERSION} or newer is required"
            ));
        }
        Ok(Self {
            size: std::mem::size_of::<Self>(),
            magic: NCCL_CONFIG_MAGIC,
            version: NCCL_CONFIG_ABI_VERSION,
            blocking: 0,
            legacy_tail_padding: c_int::MIN,
        })
    }
}

type GetVersion = unsafe extern "C" fn(*mut c_int) -> NcclResult;
type GetUniqueId = unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult;
type CommInitRankConfig =
    unsafe extern "C" fn(*mut NcclComm, c_int, NcclUniqueId, c_int, *mut NcclConfig) -> NcclResult;
type CommDestroy = unsafe extern "C" fn(NcclComm) -> NcclResult;
type CommAbort = unsafe extern "C" fn(NcclComm) -> NcclResult;
type CommGetAsyncError = unsafe extern "C" fn(NcclComm, *mut NcclResult) -> NcclResult;
type GroupStart = unsafe extern "C" fn() -> NcclResult;
type GroupEnd = unsafe extern "C" fn() -> NcclResult;
type Bcast =
    unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, NcclComm, CudaStream) -> NcclResult;
type GetErrorString = unsafe extern "C" fn(NcclResult) -> *const c_char;

struct NcclLibrary {
    _library: Library,
    get_version: GetVersion,
    get_unique_id: GetUniqueId,
    comm_init_rank_config: CommInitRankConfig,
    comm_destroy: CommDestroy,
    comm_abort: CommAbort,
    comm_get_async_error: CommGetAsyncError,
    group_start: GroupStart,
    group_end: GroupEnd,
    bcast: Bcast,
    get_error_string: GetErrorString,
}

static NCCL: OnceLock<Result<NcclLibrary, String>> = OnceLock::new();

fn library() -> Result<&'static NcclLibrary> {
    match NCCL.get_or_init(|| load_library().map_err(|error| error.to_string())) {
        Ok(library) => Ok(library),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

fn load_library() -> Result<NcclLibrary> {
    let candidates = library_candidates();
    let mut failures = Vec::new();
    for candidate in &candidates {
        let library = match unsafe { Library::new(candidate) } {
            Ok(library) => library,
            Err(error) => {
                failures.push(format!("{}: {error}", candidate.display()));
                continue;
            }
        };

        match unsafe { NcclLibrary::from_library(library) } {
            Ok(library) => return Ok(library),
            Err(error) => failures.push(format!("{}: {error}", candidate.display())),
        }
    }

    Err(anyhow!(
        "unable to load NCCL; tried [{}]. Set KVBM_NCCL_LIBRARY to libnccl.so.2. Errors: {}",
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        failures.join("; ")
    ))
}

impl NcclLibrary {
    unsafe fn from_library(library: Library) -> Result<Self> {
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let symbol = unsafe { library.get::<$ty>($name) }?;
                *symbol
            }};
        }

        Ok(Self {
            get_version: symbol!(b"ncclGetVersion\0", GetVersion),
            get_unique_id: symbol!(b"ncclGetUniqueId\0", GetUniqueId),
            comm_init_rank_config: symbol!(b"ncclCommInitRankConfig\0", CommInitRankConfig),
            comm_destroy: symbol!(b"ncclCommDestroy\0", CommDestroy),
            comm_abort: symbol!(b"ncclCommAbort\0", CommAbort),
            comm_get_async_error: symbol!(b"ncclCommGetAsyncError\0", CommGetAsyncError),
            group_start: symbol!(b"ncclGroupStart\0", GroupStart),
            group_end: symbol!(b"ncclGroupEnd\0", GroupEnd),
            bcast: symbol!(b"ncclBcast\0", Bcast),
            get_error_string: symbol!(b"ncclGetErrorString\0", GetErrorString),
            _library: library,
        })
    }
}

fn library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("KVBM_NCCL_LIBRARY") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(path) = loaded_nccl_path() {
        candidates.push(path);
    }
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
        candidates.extend(venv_nccl_candidates(Path::new(&venv)));
    }
    for variable in ["NCCL_LIB_DIR", "LD_LIBRARY_PATH"] {
        if let Some(paths) = std::env::var_os(variable) {
            for directory in std::env::split_paths(&paths) {
                candidates.push(directory.join("libnccl.so.2"));
                candidates.push(directory.join("libnccl.so"));
            }
        }
    }
    candidates.extend([PathBuf::from("libnccl.so.2"), PathBuf::from("libnccl.so")]);
    deduplicate_paths(&mut candidates);
    candidates
}

fn venv_nccl_candidates(venv: &Path) -> Vec<PathBuf> {
    let Ok(python_dirs) = std::fs::read_dir(venv.join("lib")) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    for entry in python_dirs.flatten() {
        if !entry.file_name().to_string_lossy().starts_with("python") {
            continue;
        }
        let directory = entry.path().join("site-packages/nvidia/nccl/lib");
        for name in ["libnccl.so.2", "libnccl.so"] {
            let candidate = directory.join(name);
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

fn deduplicate_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = HashSet::new();
    paths.retain(|path| seen.insert(path.clone()));
}

fn loaded_nccl_path() -> Option<PathBuf> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines().find_map(|line| {
        let path = line.split_whitespace().last()?;
        (path.contains("/libnccl.so") && Path::new(path).is_absolute()).then(|| PathBuf::from(path))
    })
}

pub(super) fn get_unique_id(output: *mut NcclUniqueId) -> Result<NcclResult> {
    Ok(unsafe { (library()?.get_unique_id)(output) })
}

pub(super) fn get_version(output: *mut c_int) -> Result<NcclResult> {
    Ok(unsafe { (library()?.get_version)(output) })
}

pub(super) fn comm_init_rank_config(
    output: *mut NcclComm,
    world_size: c_int,
    id: NcclUniqueId,
    rank: c_int,
    config: *mut NcclConfig,
) -> Result<NcclResult> {
    Ok(unsafe { (library()?.comm_init_rank_config)(output, world_size, id, rank, config) })
}

pub(super) fn comm_destroy(comm: NcclComm) -> Result<NcclResult> {
    Ok(unsafe { (library()?.comm_destroy)(comm) })
}

pub(super) fn comm_abort(comm: NcclComm) -> Result<NcclResult> {
    Ok(unsafe { (library()?.comm_abort)(comm) })
}

pub(super) fn comm_get_async_error(comm: NcclComm, output: *mut NcclResult) -> Result<NcclResult> {
    Ok(unsafe { (library()?.comm_get_async_error)(comm, output) })
}

pub(super) fn group_start() -> Result<NcclResult> {
    Ok(unsafe { (library()?.group_start)() })
}

pub(super) fn group_end() -> Result<NcclResult> {
    Ok(unsafe { (library()?.group_end)() })
}

pub(super) fn bcast(
    buffer: *mut c_void,
    count: usize,
    datatype: c_int,
    root: c_int,
    comm: NcclComm,
    stream: CudaStream,
) -> Result<NcclResult> {
    Ok(unsafe { (library()?.bcast)(buffer, count, datatype, root, comm, stream) })
}

pub(super) fn error_string(result: NcclResult) -> String {
    let Ok(library) = library() else {
        return format!("NCCL error {result}");
    };
    let ptr = unsafe { (library.get_error_string)(result) };
    if ptr.is_null() {
        format!("NCCL error {result}")
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonblocking_config_uses_exact_nccl_21400_abi() {
        let config = NcclConfig::nonblocking(23005).unwrap();

        assert_eq!(std::mem::size_of::<NcclConfig>(), 24);
        assert_eq!(std::mem::align_of::<NcclConfig>(), 8);
        assert_eq!(std::mem::offset_of!(NcclConfig, size), 0);
        assert_eq!(std::mem::offset_of!(NcclConfig, magic), 8);
        assert_eq!(std::mem::offset_of!(NcclConfig, version), 12);
        assert_eq!(std::mem::offset_of!(NcclConfig, blocking), 16);
        assert_eq!(std::mem::offset_of!(NcclConfig, legacy_tail_padding), 20);
        assert_eq!(config.size, 24);
        assert_eq!(config.magic, NCCL_CONFIG_MAGIC);
        assert_eq!(config.version, NCCL_CONFIG_ABI_VERSION);
        assert_eq!(config.blocking, 0);
        assert_eq!(config.legacy_tail_padding, c_int::MIN);
    }

    #[test]
    fn nonblocking_config_rejects_runtime_before_nccl_21400() {
        let error = NcclConfig::nonblocking(21399).err().unwrap();

        assert!(error.to_string().contains("21400 or newer is required"));
    }

    #[test]
    fn path_deduplication_preserves_explicit_priority() {
        let explicit = PathBuf::from("/explicit/libnccl.so.2");
        let loaded = PathBuf::from("/loaded/libnccl.so.2");
        let mut paths = vec![
            explicit.clone(),
            loaded.clone(),
            explicit,
            PathBuf::from("libnccl.so.2"),
        ];

        deduplicate_paths(&mut paths);

        assert_eq!(
            paths,
            vec![
                PathBuf::from("/explicit/libnccl.so.2"),
                loaded,
                PathBuf::from("libnccl.so.2"),
            ]
        );
    }

    #[test]
    fn active_python_environment_exposes_packaged_nccl() {
        let venv = tempfile::tempdir().unwrap();
        let nccl_dir = venv
            .path()
            .join("lib/python3.12/site-packages/nvidia/nccl/lib");
        std::fs::create_dir_all(&nccl_dir).unwrap();
        let versioned = nccl_dir.join("libnccl.so.2");
        std::fs::write(&versioned, []).unwrap();

        assert_eq!(venv_nccl_candidates(venv.path()), vec![versioned]);
    }
}
