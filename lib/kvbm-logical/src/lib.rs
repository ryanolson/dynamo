// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod blocks;
pub mod events;
pub mod manager;
pub mod metrics;
pub mod pools;
pub mod pubsub;
pub mod registry;
pub mod tinylfu;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

// Re-export common types and traits
pub use blocks::{
    Block, BlockError, BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock, WeakBlock,
    state::Reset,
};
pub use manager::BlockManager;
pub use pools::{BlockAllocator, DequeBlockAllocator};
pub use registry::BlockRegistry;

pub type BlockId = usize;
pub type SequenceHash = dynamo_tokens::PositionalSequenceHash;

pub trait KvbmSequenceHashProvider {
    fn kvbm_sequence_hash(&self) -> SequenceHash;
}

impl KvbmSequenceHashProvider for dynamo_tokens::TokenBlock {
    fn kvbm_sequence_hash(&self) -> SequenceHash {
        self.positional_sequence_hash()
    }
}
