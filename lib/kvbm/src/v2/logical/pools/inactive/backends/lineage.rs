// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_tokens::PositionalLineageHash;
use lru::LruCache;

use super::super::{Block, BlockMetadata, Registered};

/// A node in the lineage graph.
struct LineageNode<T: BlockMetadata> {
    /// The block stored at this node, if any.
    block: Option<Block<T, Registered>>,

    /// The sequence hash fragment of this node (redundant but useful).
    #[allow(dead_code)]
    fragment: u64,

    /// The position of this node.
    #[allow(dead_code)]
    position: u64,

    /// The parent fragment (at position - 1), if any.
    parent_fragment: Option<u64>,

    /// Children fragments (at position + 1).
    children: HashSet<u64>,
}

impl<T: BlockMetadata> LineageNode<T> {
    fn new(block: Block<T, Registered>, lineage_hash: PositionalLineageHash) -> Self {
        let parent_fragment = if lineage_hash.position() > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };

        Self {
            block: Some(block),
            fragment: lineage_hash.current_hash_fragment(),
            position: lineage_hash.position(),
            parent_fragment,
            children: HashSet::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// A backend that manages blocks using a lineage graph and evicts from the leaves.
pub struct LineageBackend<T: BlockMetadata> {
    /// Map from (position, fragment) to Node.
    /// We use a nested HashMap approach: Position -> Fragment -> Node.
    nodes: HashMap<u64, HashMap<u64, LineageNode<T>>>,

    /// LRU cache storing ONLY the keys (position, fragment) of leaf nodes.
    /// This is used to select candidates for eviction/allocation.
    leaf_lru: LruCache<(u64, u64), ()>,

    /// Total number of blocks currently stored (excluding ghost nodes).
    count: usize,

    /// Maximum capacity (total blocks).
    capacity: usize,
}

impl<T: BlockMetadata> LineageBackend<T> {
    /// Creates a new LineageBackend.
    pub fn new(capacity: std::num::NonZeroUsize) -> Self {
        // leaf_lru uses unbounded capacity because we manually manage eviction
        // based on total block count, not leaf count.
        Self {
            nodes: HashMap::new(),
            leaf_lru: LruCache::unbounded(),
            count: 0,
            capacity: capacity.get(),
        }
    }

    /// Inserts a block into the lineage graph.
    /// If capacity is exceeded, evicts the least recently used leaf.
    pub fn insert(&mut self, block: Block<T, Registered>, lineage_hash: PositionalLineageHash) {
        // Enforce capacity before insertion
        if self.count >= self.capacity {
            self.evict_leaf();
        }

        let position = lineage_hash.position();
        let fragment = lineage_hash.current_hash_fragment();
        let parent_fragment = if position > 0 {
            Some(lineage_hash.parent_hash_fragment())
        } else {
            None
        };

        // 1. Create or update the node
        let is_new_node = !self
            .nodes
            .get(&position)
            .map_or(false, |level| level.contains_key(&fragment));

        if is_new_node {
            let node = LineageNode::new(block, lineage_hash);
            self.nodes.entry(position).or_default().insert(fragment, node);
            self.count += 1;
        } else {
            // Node exists
            let level = self.nodes.get_mut(&position).unwrap();
            let node = level.get_mut(&fragment).unwrap();

            if node.block.is_none() {
                self.count += 1;
            }
            node.block = Some(block);
            node.parent_fragment = parent_fragment;
        }

        // 2. Link to parent
        if let Some(p_frag) = parent_fragment {
            let p_pos = position - 1;

            let parent_level = self.nodes.entry(p_pos).or_default();
            let parent_node = parent_level.entry(p_frag).or_insert_with(|| {
                LineageNode {
                    block: None, // Ghost node
                    fragment: p_frag,
                    position: p_pos,
                    parent_fragment: None, // We don't know the parent's parent yet
                    children: HashSet::new(),
                }
            });

            let was_parent_leaf = parent_node.is_leaf();
            parent_node.children.insert(fragment);

            // If parent was a leaf and in LRU, it is no longer a leaf. Remove from LRU.
            if was_parent_leaf {
                self.leaf_lru.pop(&(p_pos, p_frag));
            }
        }

        // 3. Update LRU status for this node
        let is_leaf = self.nodes.get(&position).unwrap().get(&fragment).unwrap().is_leaf();

        if is_leaf {
             self.leaf_lru.put((position, fragment), ());
        }
    }

    /// Evicts the least recently used leaf to free up space.
    fn evict_leaf(&mut self) {
        if let Some(((pos, frag), _)) = self.leaf_lru.pop_lru() {
            self.remove_block(pos, frag);
        }
    }

    /// Allocates (removes) a block from the pool, preferring leaves in LRU order.
    pub fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        while allocated.len() < count {
            if let Some(((pos, frag), _)) = self.leaf_lru.pop_lru() {
                if let Some(b) = self.remove_block(pos, frag) {
                    allocated.push(b);
                }
            } else {
                break; // No more leaves
            }
        }

        allocated
    }

    /// Removes a specific block by its lineage hash (for cache hits).
    pub fn remove(&mut self, lineage_hash: &PositionalLineageHash) -> Option<Block<T, Registered>> {
        let position = lineage_hash.position();
        let fragment = lineage_hash.current_hash_fragment();

        let has_block = self.nodes.get(&position)
            .and_then(|level| level.get(&fragment))
            .map(|node| node.block.is_some())
            .unwrap_or(false);

        if !has_block {
            return None;
        }

        self.leaf_lru.pop(&(position, fragment));
        self.remove_block(position, fragment)
    }

    /// Internal method to remove a block from the graph.
    /// Returns the block if one existed at that node.
    /// Handles ghost cleanup iteratively.
    fn remove_block(&mut self, position: u64, fragment: u64) -> Option<Block<T, Registered>> {
        let node_block = {
            let level = self.nodes.get_mut(&position)?;
            let node = level.get_mut(&fragment)?;
            node.block.take()
        };

        if node_block.is_some() {
            self.count -= 1;
        }

        let mut current_pos = position;
        let mut current_frag = fragment;

        // Loop for iterative cleanup upwards
        loop {
            let mut should_remove_node = false;
            let mut parent_info = None;

            if let Some(level) = self.nodes.get(&current_pos) {
                if let Some(node) = level.get(&current_frag) {
                    if node.children.is_empty() && node.block.is_none() {
                        // It's a ghost leaf (no block, no children). Prune it.
                        should_remove_node = true;
                        parent_info = node.parent_fragment.map(|pf| (current_pos.saturating_sub(1), pf));
                    }
                }
            }

            if should_remove_node {
                if let Some(level) = self.nodes.get_mut(&current_pos) {
                    level.remove(&current_frag);
                    if level.is_empty() {
                        self.nodes.remove(&current_pos);
                    }
                }

                if let Some((p_pos, p_frag)) = parent_info {
                    let mut parent_became_leaf = false;
                    let mut parent_has_block = false;

                    if let Some(level) = self.nodes.get_mut(&p_pos) {
                        if let Some(parent) = level.get_mut(&p_frag) {
                            parent.children.remove(&current_frag);
                            if parent.children.is_empty() {
                                parent_became_leaf = true;
                                parent_has_block = parent.block.is_some();
                            }
                        }
                    }

                    if parent_became_leaf {
                        if parent_has_block {
                            self.leaf_lru.put((p_pos, p_frag), ());
                            break;
                        } else {
                            current_pos = p_pos;
                            current_frag = p_frag;
                            continue;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        node_block
    }

    pub fn len(&self) -> usize {
        self.count
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    // For debugging/testing
    #[allow(dead_code)]
    pub fn get_lru_len(&self) -> usize {
        self.leaf_lru.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::logical::blocks::{Block, BlockRegistry};
    use dynamo_tokens::{PositionalLineageHash, TokenBlockSequence};
    use crate::v2::SequenceHash;
    use std::num::NonZeroUsize;

    #[derive(Clone, Debug, PartialEq)]
    struct TestData;

    // Helper to create a dummy registered block
    fn create_block(id: usize) -> Block<TestData, Registered> {
        let registry = BlockRegistry::new();
        let seq_hash = SequenceHash::default();
        let handle = registry.register_sequence_hash(seq_hash);

        let block = Block::new(id, 1);
        let tokens = vec![1u32];
        let binding = TokenBlockSequence::from_slice(&tokens, 1, None);
        let token_block = binding.blocks().into_iter().next().unwrap();

        let completed = block.complete(token_block.clone()).unwrap();
        completed.register(handle)
    }

    fn make_hash(pos: u64, current: u64, parent: u64) -> PositionalLineageHash {
         PositionalLineageHash::new(current, if pos > 0 { Some(parent) } else { None }, pos)
    }

    #[test]
    fn test_leaf_insertion() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0); // Root

        backend.insert(b1, h1);

        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_lru_len(), 1); // It is a leaf (no children)

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 1);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_parent_child_insertion() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        let b2 = create_block(2);
        let h2 = make_hash(1, 200, 100); // Child of h1

        // Insert parent first
        backend.insert(b1, h1);
        assert_eq!(backend.get_lru_len(), 1); // h1 is leaf

        // Insert child
        backend.insert(b2, h2);
        assert_eq!(backend.len(), 2);

        // h1 is no longer leaf (has child h2). h2 is leaf.
        // LRU should contain only h2.
        assert_eq!(backend.get_lru_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].block_id(), 2); // Should allocate h2 (leaf)

        // Now h1 should be a leaf again and added to LRU
        assert_eq!(backend.get_lru_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2.len(), 1);
        assert_eq!(allocated2[0].block_id(), 1);
    }

    #[test]
    fn test_out_of_order_insertion() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        let b2 = create_block(2);
        let h2 = make_hash(1, 200, 100);

        // Insert child first
        backend.insert(b2, h2);
        // Created ghost node for parent h1.
        // h2 is leaf.
        assert_eq!(backend.len(), 1); // Only 1 actual block
        assert_eq!(backend.get_lru_len(), 1);

        // Insert parent
        backend.insert(b1, h1);
        // Parent h1 fills ghost. It has child h2, so it's NOT a leaf.
        // h2 is still leaf.

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_lru_len(), 1); // Only h2

        let allocated = backend.allocate(1);
        assert_eq!(allocated[0].block_id(), 2);

        // Now h1 becomes leaf
        assert_eq!(backend.get_lru_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2[0].block_id(), 1);
    }

    #[test]
    fn test_branching() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let root = create_block(1);
        let root_hash = make_hash(0, 100, 0);

        let child1 = create_block(2);
        let child1_hash = make_hash(1, 201, 100);

        let child2 = create_block(3);
        let child2_hash = make_hash(1, 202, 100);

        backend.insert(root, root_hash);
        backend.insert(child1, child1_hash);
        backend.insert(child2, child2_hash);

        // Root has 2 children.
        // LRU should have child1 and child2. Root is not leaf.
        assert_eq!(backend.get_lru_len(), 2);

        // Allocate 2 blocks (both children)
        let allocated = backend.allocate(2);
        assert_eq!(allocated.len(), 2);

        // Now root should be leaf
        assert_eq!(backend.get_lru_len(), 1);

        let allocated_root = backend.allocate(1);
        assert_eq!(allocated_root[0].block_id(), 1);
    }

    #[test]
    fn test_chain_eviction() {
        // Chain: A -> B -> C -> D
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let blocks: Vec<_> = (0..4).map(|i| create_block(i)).collect();
        let hashes: Vec<_> = (0..4).map(|i| make_hash(i as u64, 100 + i as u64, 100 + i as u64 - 1)).collect();

        for (b, h) in blocks.into_iter().zip(hashes.into_iter()) {
            backend.insert(b, h);
        }

        assert_eq!(backend.len(), 4);
        assert_eq!(backend.get_lru_len(), 1); // Only D is leaf

        let allocated = backend.allocate(4);
        // Expect order: D, C, B, A (IDs: 3, 2, 1, 0)
        assert_eq!(allocated.len(), 4);
        assert_eq!(allocated[0].block_id(), 3);
        assert_eq!(allocated[1].block_id(), 2);
        assert_eq!(allocated[2].block_id(), 1);
        assert_eq!(allocated[3].block_id(), 0);
    }

    #[test]
    fn test_capacity_eviction() {
        // Capacity 2
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(2).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        let b2 = create_block(2);
        let h2 = make_hash(1, 200, 100); // Child of h1

        backend.insert(b1, h1);
        backend.insert(b2, h2);

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_lru_len(), 1); // Only h2 is leaf

        // Insert 3rd block (unrelated)
        let b3 = create_block(3);
        let h3 = make_hash(0, 300, 0);

        backend.insert(b3, h3);

        // Should have evicted h2 (leaf).
        // Then h1 became leaf and added to LRU.
        // Then h3 added as leaf.

        assert_eq!(backend.len(), 2); // Capacity maintained

        let allocated = backend.allocate(2);
        // Should get h1 and h3.
        let ids: Vec<_> = allocated.iter().map(|b| b.block_id()).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_remove_by_hash() {
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(10).unwrap());

        let b1 = create_block(1);
        let h1 = make_hash(0, 100, 0);

        backend.insert(b1, h1);
        assert_eq!(backend.len(), 1);

        let removed = backend.remove(&h1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().block_id(), 1);
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn test_deep_chain_cleanup_iterative() {
        // Create deep chain: 0 -> 1 -> 2 ... -> 1000
        let depth = 1000;
        let mut backend = LineageBackend::<TestData>::new(NonZeroUsize::new(2000).unwrap());

        for i in 0..depth {
            let b = create_block(i);
            let h = make_hash(i as u64, 100 + i as u64, if i > 0 { 100 + i as u64 - 1 } else { 0 });
            backend.insert(b, h);
        }

        assert_eq!(backend.len(), depth);
        // Only last one is leaf
        assert_eq!(backend.get_lru_len(), 1);

        let last_h = make_hash((depth-1) as u64, 100 + (depth-1) as u64, 100 + (depth-2) as u64);
        backend.remove(&last_h);

        assert_eq!(backend.len(), depth - 1);
        // Now 998 is leaf
        assert_eq!(backend.get_lru_len(), 1);

        // Now insert a chain out of order to create ghosts, then delete leaf to trigger cleanup
        backend = LineageBackend::<TestData>::new(NonZeroUsize::new(2000).unwrap());

        let leaf_idx = 100;
        let b_leaf = create_block(leaf_idx);
        let h_leaf = make_hash(leaf_idx as u64, 200 + leaf_idx as u64, 200 + leaf_idx as u64 - 1);

        // Insert leaf at depth 100. This creates 100 ghost parents.
        backend.insert(b_leaf, h_leaf);

        assert_eq!(backend.len(), 1); // Only 1 real block
        // Ghost nodes exist but are not counted in len

        // Remove leaf. This should iteratively clean up all 100 ghosts.
        backend.remove(&h_leaf);

        assert_eq!(backend.len(), 0);
        assert!(backend.nodes.is_empty());
    }
}
