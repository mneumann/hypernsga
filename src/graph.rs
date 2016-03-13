use graph_neighbor_matching::graph::{OwnedGraph};
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

pub fn load_graph_normalized<N>(graph_file: &str) -> OwnedGraph<N>
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
