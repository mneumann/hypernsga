use substrate::{Position, Node};
use neuron::Neuron;
use std::marker::PhantomData;
use graph_neighbor_matching::graph::{GraphBuilder, OwnedGraph};
use closed01::Closed01;

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

pub struct NeuronNetworkBuilder<P> where P: Position {
    builder: GraphBuilder<usize, Neuron>,
    _phantom: PhantomData<P>,
}

impl<P: Position> NetworkBuilder for NeuronNetworkBuilder<P> {
    type POS = P;
    type NT = Neuron;
    type Output = OwnedGraph<Neuron>;

    fn new() -> Self {
        NeuronNetworkBuilder {
            builder: GraphBuilder::new(),
            _phantom: PhantomData,
        }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, _param: f64) {
        let _ = self.builder.add_node(node.index, node.node_info.clone()); 
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let w = weight1.abs();
        debug_assert!(w <= 1.0);
        let _ = self.builder.add_edge(source_node.index, target_node.index, Closed01::new(w as f32)); 

    }
    fn network(self) -> Self::Output {
        self.builder.graph()
    }
}
