extern crate hypernsga;
extern crate nsga2;
extern crate rand;

use hypernsga::graph;
use hypernsga::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use hypernsga::cppn::{CppnDriver, GeometricActivationFunction};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Substrate, Position3d, NodeConnectivity};
use hypernsga::fitness::FitnessDomination;
use hypernsga::distribute::DistributeInterval;
use nsga2::driver::{Driver, DriverConfig};
use std::marker::PhantomData;
use std::f64::INFINITY;

fn main() {
    let mut rng = rand::thread_rng();

    let target_opt = GraphSimilarity {
        target_graph: graph::load_graph_normalized("nets/skorpion.gml"),
        edge_score: true,
        iters: 100,
        eps: 0.01,
    };

    // XXX
    let mut substrate: Substrate<Position3d, Neuron> = Substrate::new();
    let node_count = target_opt.target_graph_node_count();

    let min = -1.0;
    let max = 1.0;

    let mut z_iter = DistributeInterval::new(3, min, max); // 3 layers (Input, Hidden, Output)

    // Input layer
    {
        let z = z_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.inputs, min, max) {
            substrate.add_node(Position3d::new(x, 0.5, z), Neuron::Input, NodeConnectivity::Out);
        }
    }

    // Hidden
    {
        let z = z_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.hidden, min, max) {
            substrate.add_node(Position3d::new(x, 0.5, z), Neuron::Hidden, NodeConnectivity::InOut);
        }
    }

    // Outputs
    {
        let z = z_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.outputs, min, max) {
            substrate.add_node(Position3d::new(x, 0.5, z), Neuron::Output, NodeConnectivity::In);
        }
    }

    let driver_config = DriverConfig {
        mu: 100,
        lambda: 100,
        k: 2,
        ngen: 250,
        num_objectives: 3,
        parallel_weight: INFINITY,
    };

    let mut rng_fitness_domination = rand::thread_rng();
    let mut fitness_domination = FitnessDomination {
        p_domain_fitness: Prob::new(1.0),
        p_behavioral_diversity: Prob::new(1.0),
        p_connection_cost: Prob::new(0.25),
        rng: &mut rng_fitness_domination
    };

    let driver: CppnDriver<_,_,_,Neuron,NeuronNetworkBuilder<Position3d>> = CppnDriver {
        mating_method_weights: MatingMethodWeights {
            mutate_add_node: 10,
            mutate_drop_node: 1,
            mutate_modify_node: 1,
            mutate_connect: 20,
            mutate_disconnect: 5,
            mutate_weights: 100,
            crossover_weights: 0,
        },
        activation_functions: vec![
            GeometricActivationFunction::Linear,
            GeometricActivationFunction::BipolarGaussian,
            GeometricActivationFunction::BipolarSigmoid,
            GeometricActivationFunction::Sine,
        ],
        mutate_element_prob: Prob::new(0.05),
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian{sigma: 0.1},
        link_weight_range: WeightRange::bipolar(3.0),

        mutate_add_node_random_link_weight: false,
        mutate_drop_node_tournament_k: 5,
        mutate_modify_node_tournament_k: 2,
        mate_retries: 3,

        link_expression_threshold: 0.0,

        substrate_configuration: substrate.to_configuration(),
        domain_fitness: &target_opt,
        _netbuilder: PhantomData
    };

    driver.run(&mut rng, &driver_config, &mut fitness_domination, &|iteration, duration, num_solutions, population| {
        println!("iter: {} time: {}ns solutions: {}", iteration, duration, num_solutions); 
        let fitness_ary = population.fitness_to_vec();
        let best = fitness_ary.iter().max_by_key(|e| (e.domain_fitness * 1_000_000.0) as usize);
        println!("best: {:?}", best);
    });
}
