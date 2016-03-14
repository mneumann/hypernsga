// Domain: Target graph approximation

use closed01::Closed01;
use graph_neighbor_matching::graph::{GraphBuilder, OwnedGraph};
use std::marker::PhantomData;
use substrate::{Position, Node};
use network_builder::NetworkBuilder;
use fitness::DomainFitness;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, WeightedNodeColors};
use graph::NodeLabel;
use graph_neighbor_matching::NodeColorWeight;
use std::str::FromStr;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Neuron {
    Input,
    Output,
    Hidden,
}

impl NodeLabel for Neuron {
    fn node_label(&self, _idx: usize) -> Option<String> {
        match *self {
            Neuron::Input => Some("Input".to_owned()),
            Neuron::Hidden => Some("Hidden".to_owned()),
            Neuron::Output => Some("Output".to_owned()),
        }
    }
    fn node_shape(&self) -> &'static str {
        match *self {
            Neuron::Input => "circle",
            Neuron::Hidden => "box",
            Neuron::Output => "doublecircle",
        }
    }
}

impl NodeColorWeight for Neuron {
    fn node_color_weight(&self) -> f32 {
        match *self {
            Neuron::Input => 0.0,
            Neuron::Hidden => 1.0,
            Neuron::Output => 2.0,
        }
    }
}

impl FromStr for Neuron {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "input" => Ok(Neuron::Input),
            "output" => Ok(Neuron::Output),
            "hidden" => Ok(Neuron::Hidden),
            _ => Err("Invalid node type/weight"),
        }
    }
}

#[derive(Debug)]
pub struct NodeCount {
    pub inputs: usize,
    pub outputs: usize,
    pub hidden: usize,
}

impl NodeCount {
    pub fn from_graph(graph: &OwnedGraph<Neuron>) -> Self {
        let mut cnt = NodeCount {
            inputs: 0,
            outputs: 0,
            hidden: 0,
        };

        for node in graph.nodes() {
            match node.node_value() {
                &Neuron::Input => {
                    cnt.inputs += 1;
                }
                &Neuron::Output => {
                    cnt.outputs += 1;
                }
                &Neuron::Hidden => {
                    cnt.hidden += 1;
                }
            }
        }

        return cnt;
    }
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

#[derive(Debug)]
pub struct GraphSimilarity {
    pub target_graph: OwnedGraph<Neuron>,
    pub edge_score: bool,
    pub iters: usize,
    pub eps: f32,
}

impl GraphSimilarity {
    pub fn target_graph_node_count(&self) -> NodeCount {
        NodeCount::from_graph(&self.target_graph)
    }
}

impl DomainFitness<OwnedGraph<Neuron>> for GraphSimilarity {
    // A larger fitness means "better"
    fn fitness(&self, graph: OwnedGraph<Neuron>) -> f64 {
        let mut s = SimilarityMatrix::new(&graph, &self.target_graph, WeightedNodeColors);
        s.iterate(self.iters, self.eps);
        let assignment = s.optimal_node_assignment();
        let score = s.score_optimal_sum_norm(Some(&assignment), ScoreNorm::MaxDegree).get() as f64;
        if self.edge_score {
            score * s.score_outgoing_edge_weights_sum_norm(&assignment, ScoreNorm::MaxDegree).get() as f64
        } else {
            score
        }
    }
}
