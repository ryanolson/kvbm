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
pub(super) const NCCL_INT8: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct NcclUniqueId {
    pub(super) internal: [c_char; 128],
}

type GetUniqueId = unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult;
type CommInitRank = unsafe extern "C" fn(*mut NcclComm, c_int, NcclUniqueId, c_int) -> NcclResult;
type CommDestroy = unsafe extern "C" fn(NcclComm) -> NcclResult;
type GroupStart = unsafe extern "C" fn() -> NcclResult;
type GroupEnd = unsafe extern "C" fn() -> NcclResult;
type Bcast =
    unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, NcclComm, CudaStream) -> NcclResult;
type GetErrorString = unsafe extern "C" fn(NcclResult) -> *const c_char;

struct NcclLibrary {
    _library: Library,
    get_unique_id: GetUniqueId,
    comm_init_rank: CommInitRank,
    comm_destroy: CommDestroy,
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
            get_unique_id: symbol!(b"ncclGetUniqueId\0", GetUniqueId),
            comm_init_rank: symbol!(b"ncclCommInitRank\0", CommInitRank),
            comm_destroy: symbol!(b"ncclCommDestroy\0", CommDestroy),
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

pub(super) fn comm_init_rank(
    output: *mut NcclComm,
    world_size: c_int,
    id: NcclUniqueId,
    rank: c_int,
) -> Result<NcclResult> {
    Ok(unsafe { (library()?.comm_init_rank)(output, world_size, id, rank) })
}

pub(super) fn comm_destroy(comm: NcclComm) -> Result<NcclResult> {
    Ok(unsafe { (library()?.comm_destroy)(comm) })
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
}
