use substrate::{Position, Node};
use neuron::Neuron;
use std::marker::PhantomData;

/// Used to construct a network (graph) `G` from a CPPN and substrate combination.

pub trait NetworkBuilder {
    type POS: Position;
    type NT;
    type G;

    fn new() -> Self;
    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, param: f64);
    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                weight2: f64);
    fn network(self) -> Self::G;
}

pub struct NeuronNetworkBuilder<P> where P: Position {
    _phantom: PhantomData<P>,
}

impl<P: Position> NetworkBuilder for NeuronNetworkBuilder<P> {
    type POS = P;
    type NT = Neuron;
    type G = ();

    fn new() -> Self {
        NeuronNetworkBuilder {
            _phantom: PhantomData
        }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, param: f64) {
        unimplemented!()
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                weight2: f64) {
        unimplemented!()
    }
    fn network(self) -> Self::G {
        ()
    }
}
