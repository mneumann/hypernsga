use substrate::{Position, Node};

pub mod gml;
pub mod dot;
pub mod viz;

/// Used to construct a network (graph) `G` from a CPPN and substrate combination.

pub trait NetworkBuilder {
    type POS: Position;
    type NT;
    type Output;

    fn new() -> Self;
    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, param: f64);
    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                weight2: f64);
    fn network(self) -> Self::Output;
}
