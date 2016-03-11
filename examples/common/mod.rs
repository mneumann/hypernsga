use std::fs::File;
use std::io::Read;
use std::fmt::Debug;
use graph_neighbor_matching::{SimilarityMatrix, ScoreNorm, WeightedNodeColors, NodeColorWeight};
use graph_neighbor_matching::graph::{OwnedGraph, GraphBuilder};
use graph_neighbor_matching::{Graph, Edges};
use neat::genomes::acyclic_network::NodeType;
use graph_io_gml::parse_gml;
use asexp::Sexp;
use petgraph::Directed;
use petgraph::Graph as PetGraph;
use petgraph::graph::NodeIndex;
use std::f32::{INFINITY, NEG_INFINITY};
use closed01::Closed01;
use std::io::{self, Write};
use neat::genomes::acyclic_network::{Genome, NodeInnovation};
use std::collections::BTreeMap;

pub fn genome_to_network<T: NodeType>(genome: &Genome<T>) -> OwnedGraph<T> {
    let mut builder = GraphBuilder::new();

    genome.visit_nodes(|ext_id, node_type| {
        // make sure the node exists, even if there are no connection to it.
        let _ = builder.add_node(ext_id, node_type);
    });

    genome.visit_active_links(|src_id, target_id, weight| {
        builder.add_edge(src_id, target_id, Closed01::new(weight.into()));
    });

    return builder.graph();
}

pub fn genome_to_petgraph<T: NodeType>(genome: &Genome<T>) -> PetGraph<T, f32, Directed>  {
    let mut graph = PetGraph::new();
    let mut map: BTreeMap<NodeInnovation, NodeIndex> = BTreeMap::new();

    genome.visit_nodes(|ext_id, node_type| {
        // make sure the node exists, even if there are no connection to it.
        let node_idx = graph.add_node(node_type);
        map.insert(ext_id, node_idx);
    });

    genome.visit_active_links(|src_id, target_id, weight| {
        graph.add_edge(map[&src_id], map[&target_id], weight.into());
    });

    return graph;
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

pub fn load_graph<N, F>(graph_file: &str, convert_node_from_str: F) -> OwnedGraph<N>
    where N: Clone + Debug,
          F: Fn(&str) -> N
{
    info!("Loading graph file: {}", graph_file);

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = parse_gml(&graph_s,
                          &|node_sexp| -> Option<N> {
                              node_sexp.and_then(|se| {
                                  se.get_str().map(|s| convert_node_from_str(s))
                              })
                          },
                          &convert_weight)
                    .unwrap();
    let edge_range = determine_edge_value_range(&graph);
    let graph = graph.map(|_, nw| nw.clone(),
                          |_, &ew| normalize_to_closed01(ew, edge_range));

    OwnedGraph::from_petgraph(&graph)
}

pub fn write_gml<T, F>(filename: &str, graph: &OwnedGraph<T>, node_color_weight: &F)
    where T: NodeType + NodeLabel,
          F: Fn(&T) -> f32
{
    let mut file = File::create(filename).unwrap();
    to_gml(&mut file, graph, node_color_weight).unwrap();
}

pub fn to_gml<W, T, F>(wr: &mut W, graph: &OwnedGraph<T>, node_color_weight: &F) -> io::Result<()>
    where W: Write,
          T: NodeType + NodeLabel,
          F: Fn(&T) -> f32
{
    try!(writeln!(wr, "graph ["));
    try!(writeln!(wr, "  directed 1"));

    for nidx in 0..graph.num_nodes() {
        let node_type: f32 = node_color_weight(graph.node_value(nidx));
        try!(write!(wr, "  node [id {} weight {:.1}", nidx, node_type));
        if let Some(node_label) = graph.node_value(nidx).node_label(nidx) {
            try!(write!(wr, " label \"{}\"", node_label));
        }
        try!(writeln!(wr, "]"));
    }
    for nidx in 0..graph.num_nodes() {
        let edges = graph.out_edges_of(nidx);
        for eidx in 0..edges.num_edges() {
            try!(writeln!(wr,
                          "  edge [source {} target {} weight {:.2}]",
                          nidx,
                          edges.nth_edge(eidx).unwrap(),
                          edges.nth_edge_weight(eidx).unwrap().get()));
        }
    }
    try!(writeln!(wr, "]"));
    Ok(())
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

pub fn write_gml_petgraph<T, F>(filename: &str, graph: &PetGraph<T, f32, Directed>, node_color_weight: &F)
    where T: NodeType + NodeLabel,
          F: Fn(&T) -> f32
{
    let mut file = File::create(filename).unwrap();
    to_gml_petgraph(&mut file, graph, node_color_weight).unwrap();
}

pub fn to_gml_petgraph<W, T, F>(wr: &mut W, graph: &PetGraph<T, f32, Directed>, node_color_weight: &F) -> io::Result<()>
    where W: Write,
          T: NodeType + NodeLabel,
          F: Fn(&T) -> f32
{
    try!(writeln!(wr, "graph ["));
    try!(writeln!(wr, "  directed 1"));

    for (nidx, node) in graph.raw_nodes().iter().enumerate() {
        let node_type: f32 = node_color_weight(&node.weight);
        try!(write!(wr, "  node [id {} weight {:.1}", nidx, node_type));
        if let Some(node_label) = node.weight.node_label(nidx) {
            try!(write!(wr, " label \"{}\"", node_label));
        }
        try!(writeln!(wr, "]"));
    }
    for edge in graph.raw_edges() {
        try!(writeln!(wr,
                      "  edge [source {} target {} weight {:.2}]",
                      edge.source().index(),
                      edge.target().index(),
                      edge.weight));

    }
    try!(writeln!(wr, "]"));
    Ok(())
}

pub fn write_petgraph_as_dot<T, F>(filename: &str, graph: &PetGraph<T, f32, Directed>, node_color_weight: &F) -> io::Result<()>
    where T: NodeType + NodeLabel,
          F: Fn(&T) -> f32
{
    let mut file = File::create(filename).unwrap();
    let mut wr = &mut file;
    try!(writeln!(wr, "digraph {{"));

    for (nidx, node) in graph.raw_nodes().iter().enumerate() {
        let node_type: f32 = node_color_weight(&node.weight);
        try!(write!(wr, "  {} [weight={:.1}", nidx, node_type));
        try!(write!(wr, ",shape={}", node.weight.node_shape()));
        if let Some(node_label) = node.weight.node_label(nidx) {
            try!(write!(wr, ",label=\"{}\"", node_label));
        }
        try!(writeln!(wr, "];"));
    }
    for edge in graph.raw_edges() {
        try!(writeln!(wr,
                      "  {} -> {} [weight={:.2}];",
                      edge.source().index(),
                      edge.target().index(),
                      edge.weight));

    }
    try!(writeln!(wr, "}}"));
    Ok(())
}

pub trait NodeLabel {
    fn node_label(&self, _idx: usize) -> Option<String> {
        None
    }
    fn node_shape(&self) -> &'static str {
        "circle"
    }
}

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

impl NodeType for Neuron {
    fn accept_incoming_links(&self) -> bool {
        match *self {
            Neuron::Input => false,
            _ => true,
        }
    }
    fn accept_outgoing_links(&self) -> bool {
        match *self {
            Neuron::Output => false,
            _ => true,
        }
    }
}

pub fn convert_neuron_from_str(s: &str) -> Neuron {
    match s {
        "input" => Neuron::Input,
        "output" => Neuron::Output,
        "hidden" => Neuron::Hidden,
        _ => panic!("Invalid node type/weight"),
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
