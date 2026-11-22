// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]

use kvbm_memory::page_placement;

#[test]
#[ignore = "run this test alone under strace with get_mempolicy:error=EPERM:when=1"]
fn a_failed_policy_diagnostic_does_not_fault_a_page() {
    // SAFETY: the mapping and both mincore output bytes stay live for each
    // syscall. The mapping is unmapped before checking the results.
    let (result, before, after) = unsafe {
        let size = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let addr = libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_ne!(addr, libc::MAP_FAILED);
        let mut before = 0u8;
        let mut after = 0u8;
        assert_eq!(libc::mincore(addr, size, &mut before), 0);
        let result = page_placement(addr as usize);
        assert_eq!(libc::mincore(addr, size, &mut after), 0);
        assert_eq!(libc::munmap(addr, size), 0);
        (result, before, after)
    };
    assert_eq!(result.unwrap_err().raw_os_error(), Some(libc::EPERM));
    assert_eq!(before & 1, 0);
    assert_eq!(after & 1, 0, "the failed diagnostic faulted in the page");
}
