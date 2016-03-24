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
use hypernsga::cppn::{CppnDriver, GeometricActivationFunction, RandomGenomeCreator, Reproduction,
                      Expression, G, PopulationFitness};
use hypernsga::fitness::{Fitness, DomainFitness};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Substrate, SubstrateConfiguration, Position, Position3d, Position2d,
                           NodeConnectivity};
use hypernsga::distribute::DistributeInterval;
use nsga2::driver::{Driver, DriverConfig};
use nsga2::selection::SelectNSGP;
use nsga2::population::{UnratedPopulation, RatedPopulation, RankedPopulation};
use std::marker::PhantomData;
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
          if ui.collapsing_header(im_str!("EA Settings")).build() {
              // ui.slider_f32(im_str!("Link expression threshold"), )
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

    let driver_config = DriverConfig {
        mu: 100,
        lambda: 200,
        k: 2,
        ngen: 10000,
        num_objectives: 3,
        parallel_weight: INFINITY,
    };

    let selection = SelectNSGP { objective_eps: 0.01 };

    let reproduction = Reproduction {
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
        for _ in 0..driver_config.mu {
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

        rated.select(driver_config.mu,
                     driver_config.num_objectives,
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
    };

    loop {
        {
            support.render(CLEAR_COLOR, |ui| {
                gui(ui, &mut state);
            });
            let active = support.update_events();
            if !active {
                break;
            }
        }

        if state.running {
            // create next generation
            state.iteration += 1;
            let offspring = parents.reproduce(&mut rng,
                                              driver_config.lambda,
                                              driver_config.k,
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
            parents = next_gen.select(driver_config.mu,
                                      driver_config.num_objectives,
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
