extern crate hypernsga;
extern crate nsga2;
extern crate rand;
extern crate criterion_stats;

use hypernsga::graph;
use hypernsga::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use hypernsga::cppn::{CppnDriver, GeometricActivationFunction, RandomGenomeCreator};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Substrate, Position3d, Position2d, NodeConnectivity};
use hypernsga::distribute::DistributeInterval;
use nsga2::driver::{Driver, DriverConfig};
use nsga2::selection::{SelectNSGP};
use std::marker::PhantomData;
use std::f64::INFINITY;
use criterion_stats::univariate::Sample;
use std::env;

fn main() {
    let mut rng = rand::thread_rng();
    let graph_file = env::args().nth(1).unwrap();
    println!("graph: {}", graph_file);

    let target_opt = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&graph_file),
        edge_score: true,
        iters: 50,
        eps: 0.01,
    };

    // XXX
    let mut substrate: Substrate<Position2d, Neuron> = Substrate::new();
    let node_count = target_opt.target_graph_node_count();

    println!("{:?}", node_count);

    let min = -3.0;
    let max = 3.0;

    // Input layer
    {
        for x in DistributeInterval::new(node_count.inputs, min, max) {
            substrate.add_node(Position2d::new(x, 0.75), Neuron::Input, NodeConnectivity::Out);
        }
    }

    // Hidden
    {
        for x in DistributeInterval::new(node_count.hidden, min, max) {
            substrate.add_node(Position2d::new(x, 0.5), Neuron::Hidden, NodeConnectivity::InOut);
        }
    }

    // Outputs
    {
        for x in DistributeInterval::new(node_count.outputs, min, max) {
            substrate.add_node(Position2d::new(x, 0.25), Neuron::Output, NodeConnectivity::In);
        }
    }

    let driver_config = DriverConfig {
        mu: 100,
        lambda: 200,
        k: 2,
        ngen: 10000,
        num_objectives: 3,
        parallel_weight: INFINITY,
    };

    let selection = SelectNSGP {
        objective_eps: 0.01,
    };

    let random_genome_creator = RandomGenomeCreator {
        link_weight_range: WeightRange::bipolar(3.0),

        start_activation_functions: vec![
            //GeometricActivationFunction::Linear,
            GeometricActivationFunction::BipolarGaussian,
            GeometricActivationFunction::BipolarSigmoid,
            GeometricActivationFunction::Sine,
        ],
        start_connected: false,
        start_link_weight_range: WeightRange::bipolar(0.1),
        start_symmetry: vec![], //Some(3.0), None, Some(3.0)],
        start_initial_nodes: 0,
    };

    let driver: CppnDriver<_,_,_,Neuron,NeuronNetworkBuilder<Position2d>> = CppnDriver {
        mating_method_weights: MatingMethodWeights {
            mutate_add_node: 5,
            mutate_drop_node: 0,
            mutate_modify_node: 0,
            mutate_connect: 20,
            mutate_disconnect: 5,
            mutate_weights: 100,
            crossover_weights: 0,
        },
        activation_functions: vec![
            //GeometricActivationFunction::Linear,
            GeometricActivationFunction::BipolarGaussian,
            GeometricActivationFunction::BipolarSigmoid,
            GeometricActivationFunction::Sine,
        ],
        mutate_element_prob: Prob::new(0.05),
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian{sigma: 0.1},
        link_weight_range: WeightRange::bipolar(3.0),
        link_weight_creation_sigma: 0.1,

        mutate_add_node_random_link_weight: false,
        mutate_drop_node_tournament_k: 10,
        mutate_modify_node_tournament_k: 2,
        mate_retries: 100,

        link_expression_threshold: 0.01,

        substrate_configuration: substrate.to_configuration(),
        domain_fitness: &target_opt,
        _netbuilder: PhantomData,

        random_genome_creator: random_genome_creator,
    };

    driver.run(&mut rng, &driver_config, &selection, &|iteration, duration, num_solutions, population| {
        println!("iter: {} time: {}ms solutions: {}", iteration, duration / 1000_000, num_solutions);
        let behavioral: Vec<_> = population.individuals().iter().map(|ind| ind.fitness().behavioral_diversity as f64).collect();
        let stat = Sample::new(&behavioral);
        println!("max: {:?}", stat.max());
        println!("min: {:?}", stat.min());
        println!("mean: {:?}", stat.mean());
        // change in stddev is important, this means increase in beh.diversity 
        println!("stddev: {:?}", stat.std_dev(None));
        println!("var: {:?}", stat.var(None));

        let best = population.individuals().iter().max_by_key(|ind| (ind.fitness().domain_fitness * 1_000_000.0) as usize);
        //let worst = fitness_ary.iter().min_by_key(|e| (e.domain_fitness * 1_000_000.0) as usize);
        //println!("best: {:?} (worst: {:?})", best.unwrap().domain_fitness, worst.unwrap().domain_fitness);
        println!("best: {:?}", best.unwrap().fitness().domain_fitness);
        println!("pop-size: {}", population.len());
    });
}
