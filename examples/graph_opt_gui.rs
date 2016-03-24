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
use hypernsga::network_builder::NetworkBuilder;
use hypernsga::cppn::{GeometricActivationFunction, RandomGenomeCreator, Reproduction,
Expression, G, PopulationFitness};
use hypernsga::fitness::{Fitness, DomainFitness};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Substrate, SubstrateConfiguration, Position, Position3d, Position2d,
NodeConnectivity};
use hypernsga::distribute::DistributeInterval;
use nsga2::selection::SelectNSGP;
use nsga2::population::{UnratedPopulation, RatedPopulation, RankedPopulation};
use std::f64::INFINITY;
use criterion_stats::univariate::Sample;
use std::env;

use imgui::*;
use self::support::Support;

mod support;

const CLEAR_COLOR: (f32, f32, f32, f32) = (1.0, 1.0, 1.0, 1.0);

struct State {
    iteration: usize,
    best_fitness: f64,
    running: bool,
    mu: i32,
    lambda: i32,
    k: i32,
    mutate_add_node: i32,
    mutate_drop_node: i32,
    mutate_modify_node: i32,
    mutate_connect: i32,
    mutate_disconnect: i32,
    mutate_weights: i32,
    //crossover_weights: 0,
}

struct EvoConfig {
    mu: usize,
    lambda: usize,
    k: usize,
    num_objectives: usize,
}

fn gui<'a>(ui: &Ui<'a>, state: &mut State) {
    ui.window(im_str!("Evolutionary Graph Optimization"))
        .size((300.0, 100.0), ImGuiSetCond_FirstUseEver)
        .build(|| {
            if ui.collapsing_header(im_str!("General")).build() {
                ui.text(im_str!("Iteration: {}", state.iteration));
                ui.text(im_str!("Best Fitness: {:?}", state.best_fitness));
                ui.separator();
                if state.running {
                    if ui.small_button(im_str!("STOP")) {
                        state.running = false;
                    }
                } else {
                    if ui.small_button(im_str!("START")) {
                        state.running = true;
                    }
                }
            }
            if ui.collapsing_header(im_str!("Population Settings")).build() {
                ui.slider_i32(im_str!("Population Size"), &mut state.mu, state.k, 1000).build();
                ui.slider_i32(im_str!("Offspring Size"), &mut state.lambda, 1, 1000).build();
                ui.slider_i32(im_str!("Tournament Size"), &mut state.k, 1, state.mu).build();
            }
            if ui.collapsing_header(im_str!("Mutate Probability")).build() {
                ui.slider_i32(im_str!("Weights"), &mut state.mutate_weights, 1, 1000).build();
                ui.slider_i32(im_str!("Add Node"), &mut state.mutate_add_node, 0, 1000).build();
                ui.slider_i32(im_str!("Drop Node"), &mut state.mutate_drop_node, 0, 1000).build();
                ui.slider_i32(im_str!("Modify Node"), &mut state.mutate_modify_node, 0, 1000).build();
                ui.slider_i32(im_str!("Connect"), &mut state.mutate_connect, 0, 1000).build();
                ui.slider_i32(im_str!("Disconnect"), &mut state.mutate_disconnect, 0, 1000).build();
            }

            // ui.separator();
            // let mouse_pos = ui.imgui().mouse_pos();
            // ui.text(im_str!("Mouse Position: ({:.1},{:.1})", mouse_pos.0, mouse_pos.1));
        })
}

fn fitness<'a, P>(genome: &G,
                  expression: &Expression,
                  substrate_config: &SubstrateConfiguration<'a, P, Neuron>,
                  fitness_eval: &GraphSimilarity)
-> Fitness
where P: Position
{

    let mut network_builder = NeuronNetworkBuilder::new();
    let (behavior, connection_cost) = expression.express(genome,
                                                         &mut network_builder,
                                                         substrate_config);

    // Evaluate domain specific fitness
    let domain_fitness = fitness_eval.fitness(network_builder.network());

    Fitness {
        domain_fitness: domain_fitness,
        behavioral_diversity: 0.0, // will be calculated in `population_metric`
        connection_cost: connection_cost,
        behavior: behavior,
    }
}

fn main() {
    let mut support = Support::init();

    let mut rng = rand::thread_rng();
    let graph_file = env::args().nth(1).unwrap();
    println!("graph: {}", graph_file);

    let domain_fitness_eval = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&graph_file),
        edge_score: true,
        iters: 50,
        eps: 0.01,
    };

    // XXX
    let mut substrate: Substrate<Position2d, Neuron> = Substrate::new();
    let node_count = domain_fitness_eval.target_graph_node_count();

    println!("{:?}", node_count);

    let min = -3.0;
    let max = 3.0;

    // Input layer
    {
        for x in DistributeInterval::new(node_count.inputs, min, max) {
            substrate.add_node(Position2d::new(x, 0.75),
            Neuron::Input,
            NodeConnectivity::Out);
        }
    }

    // Hidden
    {
        for x in DistributeInterval::new(node_count.hidden, min, max) {
            substrate.add_node(Position2d::new(x, 0.5),
            Neuron::Hidden,
            NodeConnectivity::InOut);
        }
    }

    // Outputs
    {
        for x in DistributeInterval::new(node_count.outputs, min, max) {
            substrate.add_node(Position2d::new(x, 0.25),
            Neuron::Output,
            NodeConnectivity::In);
        }
    }

    let mut evo_config = EvoConfig {
        mu: 100,
        lambda: 200,
        k: 2,
        num_objectives: 3,
    };

    let selection = SelectNSGP { objective_eps: 0.01 };

    let mut reproduction = Reproduction {
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
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian { sigma: 0.1 },
        link_weight_range: WeightRange::bipolar(3.0),
        link_weight_creation_sigma: 0.1,

        mutate_add_node_random_link_weight: false,
        mutate_drop_node_tournament_k: 10,
        mutate_modify_node_tournament_k: 2,
        mate_retries: 100,
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
        start_symmetry: vec![], // Some(3.0), None, Some(3.0)],
        start_initial_nodes: 0,
    };

    let expression = Expression { link_expression_threshold: 0.01 };

    let substrate_config = substrate.to_configuration();

    // create `generation 0`
    let mut parents = {
        let mut initial = UnratedPopulation::new();
        for _ in 0..evo_config.mu {
            initial.push(random_genome_creator.create::<_, Position2d>(&mut rng));
        }
        let mut rated = initial.rate_in_parallel(&|ind| {
            fitness(ind,
                    &expression,
                    &substrate_config,
                    &domain_fitness_eval)
        },
        INFINITY);

        PopulationFitness.apply(&mut rated);

        rated.select(evo_config.mu,
                     evo_config.num_objectives,
                     &selection,
                     &mut rng)
    };

    let best_fitness = {
        let best_individual = parents.individuals().iter().max_by_key(|ind| {
            (ind.fitness().domain_fitness * 1_000_000.0) as usize
        });
        best_individual.map(|ind| ind.fitness().domain_fitness).unwrap_or(0.0)
    };

    let mut state = State {
        running: false,
        iteration: 0,
        best_fitness: best_fitness,
        mu: evo_config.mu as i32,
        lambda: evo_config.lambda as i32,
        k: evo_config.k as i32,

        mutate_add_node: reproduction.mating_method_weights.mutate_add_node as i32,
        mutate_drop_node: reproduction.mating_method_weights.mutate_drop_node as i32,
        mutate_modify_node: reproduction.mating_method_weights.mutate_modify_node as i32,
        mutate_connect: reproduction.mating_method_weights.mutate_connect as i32,
        mutate_disconnect: reproduction.mating_method_weights.mutate_disconnect as i32,
        mutate_weights: reproduction.mating_method_weights.mutate_weights as i32,
    };

    loop {
        {
            support.render(CLEAR_COLOR, |ui| {
                gui(ui, &mut state);
            });

            evo_config.mu = state.mu as usize;
            evo_config.lambda = state.lambda as usize;
            evo_config.k = state.k as usize;
            reproduction.mating_method_weights.mutate_add_node = state.mutate_add_node as u32;
            reproduction.mating_method_weights.mutate_drop_node = state.mutate_drop_node as u32;
            reproduction.mating_method_weights.mutate_modify_node = state.mutate_modify_node as u32;
            reproduction.mating_method_weights.mutate_connect = state.mutate_connect as u32;
            reproduction.mating_method_weights.mutate_disconnect = state.mutate_disconnect as u32;
            reproduction.mating_method_weights.mutate_weights = state.mutate_weights as u32;

            let active = support.update_events();
            if !active {
                break;
            }
        }

        if state.running {
            // create next generation
            state.iteration += 1;
            let offspring = parents.reproduce(&mut rng,
                                              evo_config.lambda,
                                              evo_config.k,
                                              &|rng, p1, p2| reproduction.mate(rng, p1, p2));
            let rated_offspring = offspring.rate_in_parallel(&|ind| {
                fitness(ind,
                        &expression,
                        &substrate_config,
                        &domain_fitness_eval)
            },
            INFINITY);
            let mut next_gen = parents.merge(rated_offspring);
            PopulationFitness.apply(&mut next_gen);
            parents = next_gen.select(evo_config.mu,
                                      evo_config.num_objectives,
                                      &selection,
                                      &mut rng);

            let best_individual = parents.individuals().iter().max_by_key(|ind| {
                (ind.fitness().domain_fitness * 1_000_000.0) as usize
            });
            state.best_fitness = best_individual.map(|ind| ind.fitness().domain_fitness)
                .unwrap_or(0.0);
        }
    }
}
