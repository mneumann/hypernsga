extern crate hypernsga;

use hypernsga::graph;
use hypernsga::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use hypernsga::cppn::{CppnDriver, GeometricActivationFunction};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Substrate, Position3d};
use std::marker::PhantomData;

fn main() {
    let target_opt = GraphSimilarity {
        target_graph: graph::load_graph_normalized("test.gml"),
        edge_score: true,
        iters: 100,
        eps: 0.01,
    };

    let substrate: Substrate<Position3d, Neuron> = Substrate::new();
    let substrate_configuration = substrate.to_configuration();

    let driver: CppnDriver<_,_,_,Neuron,NeuronNetworkBuilder<Position3d>> = CppnDriver {
        mating_method_weights: MatingMethodWeights {
            mutate_add_node: 1,
            mutate_drop_node: 1,
            mutate_modify_node: 1,
            mutate_connect: 10,
            mutate_disconnect: 2,
            mutate_weights: 100,
            crossover_weights: 0,
        },
        activation_functions: vec![
            GeometricActivationFunction::Linear,
            GeometricActivationFunction::BipolarGaussian,
            GeometricActivationFunction::BipolarSigmoid,
            GeometricActivationFunction::Sine,
        ],
        mutate_element_prob: Prob::new(0.03),
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian{sigma: 0.1},
        link_weight_range: WeightRange::new(3.0, -3.0),

        mutate_add_node_random_link_weight: false,
        mutate_drop_node_tournament_k: 5,
        mutate_modify_node_tournament_k: 2,
        mate_retries: 3,

        link_expression_threshold: 0.0,

        substrate_configuration: substrate.to_configuration(),
        domain_fitness: &target_opt,
        _netbuilder: PhantomData
    };
}
