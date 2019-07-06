pub use cppn_ext::position::{Position, Position2d, Position3d};

/// Represents a logical node set. Each node set is represented by a bit,
/// up to 64 node sets are supported. A node can be part of up to 64
/// different node sets. The possible connections are based on this information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeSet(u64);

impl NodeSet {
    pub fn single(n: u64) -> Self {
        assert!(n < 64);
        NodeSet(1 << n)
    }

    fn has_intersection(&self, other: &Self) -> bool {
        (self.0 & other.0) != 0
    }
}

#[test]
fn test_nodeset() {
    assert_eq!(1, NodeSet::single(0).0);
    assert_eq!(2, NodeSet::single(1).0);
    assert_eq!(4, NodeSet::single(2).0);
    assert_eq!(8, NodeSet::single(3).0);
    assert_eq!(
        true,
        NodeSet::single(0).has_intersection(&NodeSet::single(0))
    );
    assert_eq!(
        false,
        NodeSet::single(0).has_intersection(&NodeSet::single(1))
    );
    assert_eq!(
        false,
        NodeSet::single(0).has_intersection(&NodeSet::single(2))
    );
    assert_eq!(
        true,
        NodeSet::single(1).has_intersection(&NodeSet::single(1))
    );
    assert_eq!(
        false,
        NodeSet::single(1).has_intersection(&NodeSet::single(0))
    );
    assert_eq!(
        false,
        NodeSet::single(1).has_intersection(&NodeSet::single(2))
    );
}

/// Represents a node in the substrate. `T` stores additional information about that node.
#[derive(Clone, Debug)]
pub struct Node<P, T>
where
    P: Position,
{
    pub index: usize,
    pub position: P,
    pub node_info: T,
    pub node_set: NodeSet,
}

impl<P, T> Node<P, T>
where
    P: Position,
{
    fn in_nodeset(&self, node_set: &NodeSet) -> bool {
        self.node_set.has_intersection(node_set)
    }
}

#[derive(Clone, Debug)]
pub struct Substrate<P, T>
where
    P: Position,
{
    nodes: Vec<Node<P, T>>,
}

impl<P, T> Substrate<P, T>
where
    P: Position,
{
    pub fn new() -> Self {
        Substrate { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, position: P, node_info: T, node_set: NodeSet) {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            index: idx,
            position: position,
            node_info: node_info,
            node_set: node_set,
        });
    }

    pub fn nodes(&self) -> &[Node<P, T>] {
        &self.nodes
    }

    /// Determines all possible node pairs.

    pub fn to_configuration(
        self,
        connect_node_sets: &[(NodeSet, NodeSet)],
    ) -> SubstrateConfiguration<P, T> {
        let mut pairs = Vec::new();

        // Connect all nodes belonging to NodeSet `src_ns` with those that belong to `tgt_ns`.
        for &(ref src_ns, ref tgt_ns) in connect_node_sets.iter() {
            for (source_idx, source) in self.nodes.iter().enumerate() {
                // Reject invalid connections.
                if source.in_nodeset(src_ns) {
                    for (target_idx, target) in self.nodes.iter().enumerate() {
                        // Reject invalid connections.
                        if target.in_nodeset(tgt_ns) {
                            pairs.push((source_idx, target_idx));
                        }
                    }
                }
            }
        }

        pairs.sort();
        pairs.dedup();

        SubstrateConfiguration {
            nodes: self.nodes,
            links: pairs,
            null_position: P::origin(), // XXX: we might want something else than origin.
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubstrateConfiguration<P, T>
where
    P: Position,
{
    nodes: Vec<Node<P, T>>,
    links: Vec<(usize, usize)>,
    null_position: P,
}

impl<P, T> SubstrateConfiguration<P, T>
where
    P: Position,
{
    pub fn nodes(&self) -> &[Node<P, T>] {
        &self.nodes
    }
    pub fn links(&self) -> &[(usize, usize)] {
        &self.links
    }

    /// This is used to determine the configuration for a node.
    /// Usually the CPPN requires coordinate-pairs when developing links.
    /// For nodes, this `null_position` is used instead of the second coordinate.

    pub fn null_position(&self) -> &P {
        &self.null_position
    }
}
