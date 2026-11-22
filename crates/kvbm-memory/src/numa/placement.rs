// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The NUMA placement decision for pinned host memory, and a reader for the
//! placement the kernel reports for a page.
//!
//! [`decide_placement`] is pure: it takes the device's node, the
//! NUMA-disabled flag and the CPU topology, and returns either a node to bind
//! or the reason the OS places the pages. [`page_placement`] asks the kernel
//! (`get_mempolicy`) what policy covers an address and which node holds its
//! page, so tests and callers can check the placement without trusting the
//! allocator's own record.

use nix::libc;
use serde::{Deserialize, Serialize};

use super::NumaNode;
use super::topology::NumaTopology;
use crate::mmap_pinned::PlacementFallback;

/// The outcome of [`decide_placement`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlacementDecision {
    /// Bind the pages to this node, from a thread pinned to it.
    Bind(NumaNode),
    /// Let the OS place the pages, for this reason.
    OsDefault(PlacementFallback),
}

/// Decide where pinned host memory for a device goes.
///
/// Binds only to a known node that has CPUs. Checks, in order: NUMA disabled,
/// node unknown, topology unavailable, node without CPUs.
pub(crate) fn decide_placement(
    device_node: Option<NumaNode>,
    numa_disabled: bool,
    topology: Result<&NumaTopology, &str>,
) -> PlacementDecision {
    let _ = (device_node, numa_disabled, topology);
    todo!("decide_placement")
}

/// A NUMA memory policy mode, as `get_mempolicy` reports it, without the
/// mode flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemPolicyMode {
    /// `MPOL_DEFAULT`: no policy on the range; the task's policy applies.
    Default,
    /// `MPOL_PREFERRED`.
    Preferred,
    /// `MPOL_BIND`.
    Bind,
    /// `MPOL_INTERLEAVE`.
    Interleave,
    /// `MPOL_LOCAL`.
    Local,
    /// Any other mode number.
    Other(i32),
}

/// What the kernel reports for the page that holds one address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagePlacement {
    /// The policy mode on the range (`get_mempolicy(MPOL_F_ADDR)`).
    pub mode: MemPolicyMode,
    /// The nodes in the policy's node mask. Empty for [`MemPolicyMode::Default`].
    pub nodes: Vec<NumaNode>,
    /// The node that holds the page
    /// (`get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR)`). The kernel faults the
    /// page in if it is not yet present.
    pub node: NumaNode,
}

/// Ask the kernel for the memory policy on `addr` and the node of its page.
///
/// A diagnostic: it reads, and changes nothing except that the kernel faults
/// in a page that is not yet present. Fails with the `get_mempolicy` error,
/// for example `EPERM` under a seccomp profile that denies the call.
pub fn page_placement(addr: usize) -> std::io::Result<PagePlacement> {
    let _ = addr;
    todo!("page_placement")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Like gb300: node 0 has CPUs, nodes 1 and 2 are memory-only.
    fn topology() -> NumaTopology {
        NumaTopology::from_node_cpus(HashMap::from([
            (0, vec![0, 1, 2, 3]),
            (1, vec![]),
            (2, vec![]),
        ]))
    }

    #[test]
    fn placement_binds_a_node_with_cpus() {
        let topo = topology();
        assert_eq!(
            decide_placement(Some(NumaNode(0)), false, Ok(&topo)),
            PlacementDecision::Bind(NumaNode(0))
        );
    }

    #[test]
    fn placement_falls_back_when_the_node_is_unknown() {
        let topo = topology();
        assert_eq!(
            decide_placement(None, false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeUnknown)
        );
        assert_eq!(
            decide_placement(Some(NumaNode::UNKNOWN), false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeUnknown)
        );
    }

    #[test]
    fn placement_refuses_a_node_without_cpus() {
        let topo = topology();
        assert_eq!(
            decide_placement(Some(NumaNode(1)), false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeHasNoCpus(NumaNode(1)))
        );
        // A node the topology does not list has no CPUs either.
        assert_eq!(
            decide_placement(Some(NumaNode(7)), false, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NodeHasNoCpus(NumaNode(7)))
        );
    }

    #[test]
    fn placement_falls_back_when_numa_is_disabled() {
        let topo = topology();
        assert_eq!(
            decide_placement(Some(NumaNode(0)), true, Ok(&topo)),
            PlacementDecision::OsDefault(PlacementFallback::NumaDisabled)
        );
    }

    #[test]
    fn placement_falls_back_without_a_topology() {
        assert_eq!(
            decide_placement(Some(NumaNode(0)), false, Err("no topology")),
            PlacementDecision::OsDefault(PlacementFallback::TopologyUnavailable)
        );
    }

    #[test]
    fn placement_fallback_serializes() {
        let f = PlacementFallback::BindRefused {
            node: NumaNode(3),
            error: "mbind: Invalid argument".into(),
        };
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<PlacementFallback>(&json).unwrap(), f);
    }
}
