use cppn_ext::position::Position;

#[derive(Debug, Copy, Clone)]
pub enum NodeConnectivity {
    In,
    Out,
    InOut
}

/// Represents a node in the substrate. `T` stores additional information about that node.
#[derive(Clone, Debug)]
pub struct Node<P, T>
    where P: Position,
{
    pub index: usize,
    pub position: P,
    pub node_info: T,
    pub node_connectivity: NodeConnectivity
}

#[derive(Clone, Debug)]
pub struct Substrate<P, T>
    where P: Position,
{
    nodes: Vec<Node<P,T>>,
}

impl<P, T> Substrate<P, T>
    where P: Position,
{
    pub fn new() -> Self {
        Substrate {
            nodes: Vec::new(),
        }
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

    pub fn nodes(&self) -> &[Node<P,T>] {
        &self.nodes
    }

    /// Determines all possible node pairs.

    pub fn allowed_node_pairs(&self) -> Vec<(&Node<P,T>, &Node<P,T>)> {
        let mut pairs = Vec::new();

        for source in self.nodes.iter() {
            // Reject invalid connections.
            match source.node_connectivity {
                NodeConnectivity::Out | NodeConnectivity::InOut => {}
                NodeConnectivity::In => {
                    // Node does not allow outgoing connections
                    continue;
                }
            }

            for target in self.nodes.iter() {
                match target.node_connectivity {
                    NodeConnectivity::In | NodeConnectivity::InOut => {
                        pairs.push((source, target));
                    }
                    NodeConnectivity::Out => {
                        // Node does not allow incoming connections
                    }
                }
            }
        }

        pairs
    }



    /// Determines the maximal number of links

    pub fn max_links(&self) -> usize {
        let mut link_count = 0;

        for source in self.nodes.iter() {
            // Reject invalid connections.
            match source.node_connectivity {
                NodeConnectivity::Out | NodeConnectivity::InOut => {}
                NodeConnectivity::In => {
                    // Node does not allow outgoing connections
                    continue;
                }
            }

            for target in self.nodes.iter() {
                match target.node_connectivity {
                    NodeConnectivity::In | NodeConnectivity::InOut => {
                        link_count += 1;
                    }
                    NodeConnectivity::Out => {
                        // Node does not allow incoming connections
                    }
                }
            }
        }

        link_count
    }

}
