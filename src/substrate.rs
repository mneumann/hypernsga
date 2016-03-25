pub use cppn_ext::position::{Position, Position3d, Position2d};

#[derive(Debug, Copy, Clone)]
pub enum NodeConnectivity {
    In,
    Out,
    InOut,
}

/// Represents a node in the substrate. `T` stores additional information about that node.
#[derive(Clone, Debug)]
pub struct Node<P, T>
    where P: Position
{
    pub index: usize,
    pub position: P,
    pub node_info: T,
    pub node_connectivity: NodeConnectivity,
}

#[derive(Clone, Debug)]
pub struct Substrate<P, T>
    where P: Position
{
    nodes: Vec<Node<P, T>>,
}

impl<P, T> Substrate<P, T> where P: Position
{
    pub fn new() -> Self {
        Substrate { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, position: P, node_info: T, node_connectivity: NodeConnectivity) {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            index: idx,
            position: position,
            node_info: node_info,
            node_connectivity: node_connectivity,
        });
    }

    pub fn nodes(&self) -> &[Node<P, T>] {
        &self.nodes
    }

    /// Determines all possible node pairs.

    pub fn to_configuration(self) -> SubstrateConfiguration<P, T> {
        let mut pairs = Vec::new();

        for (source_idx, source) in self.nodes.iter().enumerate() {
            // Reject invalid connections.
            match source.node_connectivity {
                NodeConnectivity::Out | NodeConnectivity::InOut => {}
                NodeConnectivity::In => {
                    // Node does not allow outgoing connections
                    continue;
                }
            }

            for (target_idx, target) in self.nodes.iter().enumerate() {
                match target.node_connectivity {
                    NodeConnectivity::In | NodeConnectivity::InOut => {
                        pairs.push((source_idx, target_idx));
                    }
                    NodeConnectivity::Out => {
                        // Node does not allow incoming connections
                    }
                }
            }
        }

        SubstrateConfiguration {
            nodes: self.nodes,
            links: pairs,
            null_position: P::origin(), // XXX: we might want something else than origin.
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubstrateConfiguration<P, T>
    where P: Position,
{
    nodes: Vec<Node<P, T>>,
    links: Vec<(usize, usize)>,
    null_position: P,
}

impl<P, T> SubstrateConfiguration<P, T>
    where P: Position
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
