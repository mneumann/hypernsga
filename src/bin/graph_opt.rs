extern crate hypernsga;

use hypernsga::graph;
use hypernsga::domain_graph::GraphSimilarity;

fn main() {
    let _target_opt = GraphSimilarity {
        target_graph: graph::load_graph_normalized("test.gml"),
        edge_score: true,
        iters: 100,
        eps: 0.01,
    };
}
