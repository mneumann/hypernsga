extern crate hypernsga;
extern crate nsga2;
extern crate rand;
extern crate criterion_stats;

#[macro_use]
extern crate glium;
#[macro_use]
extern crate imgui;
extern crate time;

use hypernsga::graph;
use hypernsga::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use hypernsga::cppn::{CppnDriver, GeometricActivationFunction};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Substrate, Position3d, Position2d, NodeConnectivity};
use hypernsga::distribute::DistributeInterval;
use nsga2::driver::{Driver, DriverConfig};
use nsga2::selection::{SelectNSGP};
use nsga2::population::{UnratedPopulation, RankedPopulation};
use std::marker::PhantomData;
use std::f64::INFINITY;
use criterion_stats::univariate::Sample;
use std::env;

use imgui::*;
use self::support::Support;

mod support;

const CLEAR_COLOR: (f32, f32, f32, f32) = (1.0, 1.0, 1.0, 1.0);

fn gui<'a>(ui: &Ui<'a>, iteration: usize, best_fitness: f64) {
    ui.window(im_str!("Evolutionary Graph Optimization"))
        .size((300.0, 100.0), ImGuiSetCond_FirstUseEver)
        .build(|| {
            ui.text(im_str!("Iteration: {}", iteration));
            ui.text(im_str!("Best Fitness: {:?}", best_fitness));
            ui.separator();
            let mouse_pos = ui.imgui().mouse_pos();
            ui.text(im_str!("Mouse Position: ({:.1},{:.1})", mouse_pos.0, mouse_pos.1));
        })
}


fn main() {
    let mut support = Support::init();

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

        start_connected: false,
        start_link_weight_range: WeightRange::bipolar(0.1),
        start_symmetry: vec![], //Some(3.0), None, Some(3.0)],
        start_initial_nodes: 0,
    };

    let mut generation: usize = 0;

    // create `generation 0`
    let mut parents = {
        let initial = driver.initial_population(&mut rng, driver_config.mu);
        driver.merge_and_select(driver.empty_parent_population(), initial, &mut rng, &driver_config, &selection)
    };

    loop {
        {
            let best_individual = parents.individuals().iter().max_by_key(|ind| (ind.fitness().domain_fitness * 1_000_000.0) as usize);
            let best_fitness = best_individual.map(|ind| ind.fitness().domain_fitness).unwrap_or(0.0);

            support.render(CLEAR_COLOR, |ui| {
                gui(ui, generation, best_fitness);
            });
            let active = support.update_events();
            if !active { break }
        }

        // create next generation
        generation += 1;
        let offspring = driver.reproduce(&parents, &mut rng, &driver_config);
        parents = driver.merge_and_select(parents, offspring, &mut rng, &driver_config, &selection);
    }
}
