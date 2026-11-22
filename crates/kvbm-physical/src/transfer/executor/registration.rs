use super::PhysicalLayout;
use anyhow::{Result, ensure};
use kvbm_memory::nixl::{MappedRegistrationGuard, XferDescList};
use std::{ops::Range, sync::Arc};

pub(crate) type RegistrationGuards = Vec<Arc<dyn MappedRegistrationGuard>>;

pub(crate) struct CudaSubmissionGuard {
    stream: Arc<cudarc::driver::CudaStream>,
    guards: RegistrationGuards,
}

impl CudaSubmissionGuard {
    pub(crate) fn new(stream: Arc<cudarc::driver::CudaStream>, guards: RegistrationGuards) -> Self {
        Self { stream, guards }
    }
    pub(crate) fn take(&mut self) -> RegistrationGuards {
        std::mem::take(&mut self.guards)
    }
}

impl Drop for CudaSubmissionGuard {
    fn drop(&mut self) {
        if !self.guards.is_empty() && self.stream.synchronize().is_err() {
            std::mem::forget(std::mem::take(&mut self.guards));
        }
    }
}

pub(crate) fn acquire_blocks(
    layout: &PhysicalLayout,
    blocks: &[crate::BlockId],
    layers: Option<&Range<usize>>,
) -> Result<RegistrationGuards> {
    let mut guards = Vec::new();
    if layout.registration_provider().is_none() {
        return Ok(guards);
    }
    let layers = layers.cloned().unwrap_or(0..layout.layout().num_layers());
    for &block in blocks {
        for layer in layers.clone() {
            for outer in 0..layout.layout().outer_dim() {
                let region = layout.memory_region(block, layer, outer)?;
                acquire(layout, region.addr(), region.size(), &mut guards)?;
            }
        }
    }
    Ok(guards)
}

fn acquire(
    layout: &PhysicalLayout,
    address: usize,
    bytes: usize,
    guards: &mut RegistrationGuards,
) -> Result<Vec<Range<usize>>> {
    let Some(provider) = layout.registration_provider() else {
        let end = address
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("transfer range overflow"))?;
        return Ok(std::iter::once(address..end).collect());
    };
    let lease = provider.acquire(address, bytes)?;
    validate_ranges(&lease.ranges, bytes)?;
    guards.push(lease.guard);
    Ok(lease.ranges)
}

/// A lease may give its ranges at other addresses than the requested range,
/// for example in a registered arena that maps the same memory. The ranges
/// must be in transfer order and their lengths must add up to the request.
fn validate_ranges(ranges: &[Range<usize>], bytes: usize) -> Result<()> {
    let mut covered = 0usize;
    for range in ranges {
        ensure!(
            range.end > range.start,
            "registration lease has an empty or reversed range"
        );
        covered = covered
            .checked_add(range.len())
            .filter(|&covered| covered <= bytes)
            .ok_or_else(|| {
                anyhow::anyhow!("registration lease covers more than the transfer range")
            })?;
    }
    ensure!(
        covered == bytes,
        "registration lease has incomplete coverage"
    );
    Ok(())
}

pub(crate) fn append_pair(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_address: usize,
    dst_address: usize,
    bytes: usize,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
    guards: &mut RegistrationGuards,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let source = acquire(src, src_address, bytes, guards)?;
    let destination = acquire(dst, dst_address, bytes, guards)?;
    for (source, destination, size) in split_pairs(&source, &destination) {
        src_dl.add_desc(source, size, src.nixl_metadata().device_id());
        dst_dl.add_desc(destination, size, dst.nixl_metadata().device_id());
    }
    Ok(())
}

fn split_pairs(
    source: &[Range<usize>],
    destination: &[Range<usize>],
) -> Vec<(usize, usize, usize)> {
    let mut result = Vec::new();
    let (mut si, mut di, mut so, mut offset) = (0, 0, 0, 0);
    while si < source.len() && di < destination.len() {
        let size = (source[si].len() - so).min(destination[di].len() - offset);
        result.push((source[si].start + so, destination[di].start + offset, size));
        so += size;
        offset += size;
        if so == source[si].len() {
            si += 1;
            so = 0;
        }
        if offset == destination[di].len() {
            di += 1;
            offset = 0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits_at_both_registration_boundaries() {
        assert_eq!(
            split_pairs(&[10..14, 14..22], &[30..37, 37..42]),
            vec![(10, 30, 4), (14, 34, 3), (17, 37, 5)]
        );
    }
    #[test]
    fn registration_coverage_rejects_empty_short_and_long_ranges() {
        assert!(validate_ranges(std::slice::from_ref(&(10..19)), 10).is_err());
        assert!(validate_ranges(std::slice::from_ref(&(10..21)), 10).is_err());
        assert!(validate_ranges(&[10..10, 10..20], 10).is_err());
        assert!(validate_ranges(&[10..14, 15..22], 10).is_err());
        assert!(validate_ranges(&[], 10).is_err());
        assert!(validate_ranges(&[10..14, 14..20], 10).is_ok());
    }
    #[test]
    fn registration_coverage_accepts_translated_ranges() {
        assert!(validate_ranges(&[4096..4100, 100..106], 10).is_ok());
        assert_eq!(
            split_pairs(&[4096..4100, 100..106], std::slice::from_ref(&(30..40))),
            vec![(4096, 30, 4), (100, 34, 6)]
        );
    }
}
