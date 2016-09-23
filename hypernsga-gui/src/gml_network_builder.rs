use hypernsga::network_builder::NetworkBuilder;
use hypernsga::domain_graph::Neuron;
use hypernsga::substrate::{Position3d, Node};
use std::io::Write;

pub struct GMLNetworkBuilder<'a, W: Write + 'a> {
    wr: Option<&'a mut W>,
}

impl<'a, W: Write> GMLNetworkBuilder<'a, W> {
    pub fn set_writer(&mut self, wr: &'a mut W) {
        self.wr = Some(wr);
    }
    pub fn begin(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "graph [").unwrap();
        writeln!(wr, "directed 1").unwrap();
    }
    pub fn end(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "]").unwrap();
    }
}

impl<'a, W: Write> NetworkBuilder for GMLNetworkBuilder<'a, W> {
    type POS = Position3d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        GMLNetworkBuilder { wr: None }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, _param: f64) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "  node [id {} weight {:.1}]", node.index, 0.0).unwrap();
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let wr = self.wr.as_mut().unwrap();
        let w = weight1.abs();
        debug_assert!(w <= 1.0);
        writeln!(wr,
                 "  edge [source {} target {} weight {:.1}]",
                 source_node.index,
                 target_node.index,
                 w)
            .unwrap();
    }

    fn network(self) -> Self::Output {
        ()
    }
}


