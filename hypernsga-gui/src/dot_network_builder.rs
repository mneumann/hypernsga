use hypernsga::network_builder::NetworkBuilder;
use hypernsga::domain_graph::Neuron;
use hypernsga::substrate::{Position3d, Node};
use std::io::Write;

pub struct DotNetworkBuilder<'a, W: Write + 'a> {
    wr: Option<&'a mut W>,
}

impl<'a, W: Write> DotNetworkBuilder<'a, W> {
    pub fn set_writer(&mut self, wr: &'a mut W) {
        self.wr = Some(wr);
    }
    pub fn begin(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "digraph {{").unwrap();
        writeln!(wr, "graph [layout=dot,overlap=false];").unwrap();
        writeln!(wr, "node [fontname = Helvetica];").unwrap();
    }
    pub fn end(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "}}").unwrap();
    }
}

impl<'a, W: Write> NetworkBuilder for DotNetworkBuilder<'a, W> {
    type POS = Position3d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        DotNetworkBuilder { wr: None }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, param: f64) {
        let wr = self.wr.as_mut().unwrap();
        let rank = match node.node_info {
            Neuron::Input => ",rank=min",
            Neuron::Hidden => "",
            Neuron::Output => ",rank=max",
        };
        writeln!(wr,
                 "  {}[label={},weight={:.1}{}];",
                 node.index,
                 node.index,
                 param,
                 rank)
            .unwrap();
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let wr = self.wr.as_mut().unwrap();
        let color = if weight1 >= 0.0 { "black" } else { "red" };
        let w = weight1.abs();
        // debug_assert!(w <= 1.0);
        writeln!(wr,
                 "  {} -> {} [weight={:.2},color={}];",
                 source_node.index,
                 target_node.index,
                 w,
                 color)
            .unwrap();
    }

    fn network(self) -> Self::Output {
        ()
    }
}

