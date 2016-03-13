use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, WeightedNodeColors};
use graph_neighbor_matching::graph::{OwnedGraph};
use graph_neighbor_matching::{Graph};
use neuron::Neuron;
use std::fs::File;
use std::io::Read;
use std::str::FromStr;
use graph_io_gml::parse_gml;
use std::f32::{INFINITY, NEG_INFINITY};
use closed01::Closed01;
//use std::io::{self, Write};
use asexp::Sexp;
use petgraph::Directed;
use petgraph::Graph as PetGraph;
use std::fmt::Debug;

/// Trait used for dot generation

pub trait NodeLabel {
    fn node_label(&self, _idx: usize) -> Option<String> {
        None
    }
    fn node_shape(&self) -> &'static str {
        "circle"
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
    // A larger fitness means "better"
    pub fn fitness(&self, graph: &OwnedGraph<Neuron>) -> f32 {
        let mut s = SimilarityMatrix::new(graph, &self.target_graph, WeightedNodeColors);
        s.iterate(self.iters, self.eps);
        let assignment = s.optimal_node_assignment();
        let score = s.score_optimal_sum_norm(Some(&assignment), ScoreNorm::MaxDegree).get();
        if self.edge_score {
            score * s.score_outgoing_edge_weights_sum_norm(&assignment, ScoreNorm::MaxDegree).get()
        } else {
            score
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

fn convert_weight(w: Option<&Sexp>) -> Option<f32> {
    match w {
        Some(s) => s.get_float().map(|f| f as f32),
        None => {
            // use a default
            Some(0.0)
        }
    }
}

fn determine_edge_value_range<T>(g: &PetGraph<T, f32, Directed>) -> (f32, f32) {
    let mut w_min = INFINITY;
    let mut w_max = NEG_INFINITY;
    for i in g.raw_edges() {
        w_min = w_min.min(i.weight);
        w_max = w_max.max(i.weight);
    }
    (w_min, w_max)
}

fn normalize_to_closed01(w: f32, range: (f32, f32)) -> Closed01<f32> {
    assert!(range.1 >= range.0);
    let dist = range.1 - range.0;
    if dist == 0.0 {
        Closed01::zero()
    } else {
        Closed01::new((w - range.0) / dist)
    }
}

pub fn load_graph<N, F>(graph_file: &str) -> OwnedGraph<N>
    where N: Clone + Debug + FromStr<Err = &'static str>
{
    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = parse_gml(&graph_s,
                          &|node_sexp| -> Option<N> {
                              node_sexp.and_then(|se| se.get_str().map(|s| N::from_str(s).unwrap()))
                          },
                          &convert_weight)
                    .unwrap();
    let edge_range = determine_edge_value_range(&graph);
    let graph = graph.map(|_, nw| nw.clone(),
                          |_, &ew| normalize_to_closed01(ew, edge_range));

    OwnedGraph::from_petgraph(&graph)
}
