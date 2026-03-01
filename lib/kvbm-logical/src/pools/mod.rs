// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block pool RAII guards and allocation traits for thread-safe block management.
//!
//! This module provides:
//! - Type-safe RAII guards (MutableBlock, CompleteBlock, ImmutableBlock) for automatic resource cleanup
//! - ResetPool: Pool for mutable blocks in reset state
//! - InactivePool: Pool for inactive immutable registered blocks
//! - BlockRegistry: Global registry for block deduplication via weak references
//! - Pluggable allocation and reuse policies

mod active;
mod inactive;
mod reset;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
mod block_proptest;

pub(crate) use active::ActivePool;
pub(crate) use inactive::backends;
pub(crate) use inactive::{InactivePool, InactivePoolBackend};
pub(crate) use reset::ResetPool;
pub use reset::DequeBlockAllocator;

// Re-export RAII guards from guards module
use crate::blocks::{
    Block, BlockId, BlockMetadata, ImmutableBlock, MutableBlock, PrimaryBlock, RegisteredBlock,
    state::{Registered, Reset},
};

pub(crate) use crate::SequenceHash;

/// Pluggable allocation strategy for blocks in the **Reset** state.
///
/// Implementations manage a pool of `Block<T, Reset>` values. The
/// [`ResetPool`] (and by extension [`BlockManager`](crate::BlockManager))
/// calls [`pop`](Self::pop) to hand out blocks and [`insert`](Self::insert)
/// when a block is returned via RAII drop.
///
/// The default implementation ([`DequeBlockAllocator`](super::pools::reset::DequeBlockAllocator))
/// uses a simple FIFO queue. Custom implementations can perform
/// arbitrary work on allocation and deallocation — for example,
/// communicating with a remote block server.
///
/// # Thread safety
///
/// The allocator is always accessed behind a `Mutex`, so implementations
/// do **not** need internal synchronisation.
pub trait BlockAllocator<T: BlockMetadata> {
    /// Return a block to the allocator.
    ///
    /// Called automatically by the RAII drop path of [`MutableBlock`](crate::MutableBlock)
    /// and [`CompleteBlock`](crate::CompleteBlock).
    fn insert(&mut self, block: Block<T, Reset>);

    /// Remove and return the next available block, or `None` if the
    /// allocator is empty.
    fn pop(&mut self) -> Option<Block<T, Reset>>;

    /// Number of blocks currently held by the allocator.
    fn len(&self) -> usize;

    /// Returns `true` when the allocator holds no blocks.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Blanket impl so `Box<dyn BlockAllocator<T>>` can be passed where
/// `impl BlockAllocator<T>` is expected (e.g. the builder).
impl<T: BlockMetadata> BlockAllocator<T> for Box<dyn BlockAllocator<T> + Send + Sync> {
    fn insert(&mut self, block: Block<T, Reset>) {
        (**self).insert(block)
    }

    fn pop(&mut self) -> Option<Block<T, Reset>> {
        (**self).pop()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

#[expect(dead_code)]
pub(crate) trait BlockMatcher<T: BlockMetadata> {
    fn find_match(&self, seq_hash: SequenceHash) -> Option<ImmutableBlock<T>>;
}

// Re-export block duplication policy
pub use crate::blocks::BlockDuplicationPolicy;

// Re-export reuse policy from inactive backends
pub use inactive::backends::{ReusePolicy, ReusePolicyError};

// Re-export the new RAII guard types - no need to re-export here since they're defined in this module

/// A block that is free and available for allocation
/// This block must be in a Registered state and have a valid sequence hash
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InactiveBlock {
    pub block_id: BlockId,
    pub seq_hash: SequenceHash,
}

// RegisteredPool implementation moved to registered.rs
