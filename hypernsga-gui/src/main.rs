extern crate hypernsga;
extern crate nsga2;
extern crate rand;

#[macro_use]
extern crate glium;
#[macro_use]
extern crate imgui;
extern crate time;
extern crate graph_layout;

use hypernsga::graph;
use hypernsga::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use hypernsga::network_builder::NetworkBuilder;
use hypernsga::cppn::{CppnNodeKind, ActivationFunction, GeometricActivationFunction,
                      RandomGenomeCreator, Reproduction, Expression, G, PopulationFitness};
use hypernsga::fitness::{Fitness, DomainFitness};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Node, SubstrateConfiguration, Position, Position3d};
use nsga2::selection::SelectNSGPMod;
use nsga2::population::{UnratedPopulation};
use std::f64::INFINITY;
use std::env;

use self::support::Support;
use glium::Surface;
use std::io::Write;
use std::fs::File;
use std::cmp::Ordering;
pub use vertex::Vertex;
use render_graph::{render_graph, Transformation};
use render_cppn::render_cppn;
use imgui_ui::gui;
pub use ui_state::{State, Action, ViewMode};

mod support;
mod viz_network_builder;
mod vertex;
mod shaders;
mod substrate_configuration;
mod render_graph;
mod render_cppn;
mod imgui_ui;
mod ui_state;

const CLEAR_COLOR: (f32, f32, f32, f32) = (1.0, 1.0, 1.0, 1.0);

pub struct GMLNetworkBuilder<'a, W: Write + 'a> {
    wr: Option<&'a mut W>,
}

impl<'a, W: Write> GMLNetworkBuilder<'a, W> {
    fn set_writer(&mut self, wr: &'a mut W) {
        self.wr = Some(wr);
    }
    fn begin(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "graph [").unwrap();
        writeln!(wr, "directed 1").unwrap();
    }
    fn end(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "]").unwrap();
    }
}

impl<'a, W: Write> NetworkBuilder for GMLNetworkBuilder<'a, W> {
    type POS = Position3d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        GMLNetworkBuilder { wr: None }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, _param: f64) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "  node [id {} weight {:.1}]", node.index, 0.0).unwrap();
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let wr = self.wr.as_mut().unwrap();
        let w = weight1.abs();
        debug_assert!(w <= 1.0);
        writeln!(wr,
                 "  edge [source {} target {} weight {:.1}]",
                 source_node.index,
                 target_node.index,
                 w)
            .unwrap();
    }

    fn network(self) -> Self::Output {
        ()
    }
}


pub struct DotNetworkBuilder<'a, W: Write + 'a> {
    wr: Option<&'a mut W>,
}

impl<'a, W: Write> DotNetworkBuilder<'a, W> {
    fn set_writer(&mut self, wr: &'a mut W) {
        self.wr = Some(wr);
    }
    fn begin(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "digraph {{").unwrap();
        writeln!(wr, "graph [layout=dot,overlap=false];").unwrap();
        writeln!(wr, "node [fontname = Helvetica];").unwrap();
    }
    fn end(&mut self) {
        let wr = self.wr.as_mut().unwrap();
        writeln!(wr, "}}").unwrap();
    }
}

impl<'a, W: Write> NetworkBuilder for DotNetworkBuilder<'a, W> {
    type POS = Position3d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        DotNetworkBuilder { wr: None }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, param: f64) {
        let wr = self.wr.as_mut().unwrap();
        let rank = match node.node_info {
            Neuron::Input => ",rank=min",
            Neuron::Hidden => "",
            Neuron::Output => ",rank=max",
        };
        writeln!(wr,
                 "  {}[label={},weight={:.1}{}];",
                 node.index,
                 node.index,
                 param,
                 rank)
            .unwrap();
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let wr = self.wr.as_mut().unwrap();
        let color = if weight1 >= 0.0 { "black" } else { "red" };
        let w = weight1.abs();
        // debug_assert!(w <= 1.0);
        writeln!(wr,
                 "  {} -> {} [weight={:.2},color={}];",
                 source_node.index,
                 target_node.index,
                 w,
                 color)
            .unwrap();
    }

    fn network(self) -> Self::Output {
        ()
    }
}

struct EvoConfig {
    mu: usize,
    lambda: usize,
    k: usize,
    objectives: Vec<usize>,
}

fn fitness<P>(genome: &G,
              expression: &Expression,
              substrate_config: &SubstrateConfiguration<P, Neuron>,
              fitness_eval: &GraphSimilarity)
              -> Fitness
    where P: Position
{

    let mut network_builder = NeuronNetworkBuilder::new();
    let (behavior, connection_cost, sat) =
        expression.express(genome, &mut network_builder, substrate_config);

    // Evaluate domain specific fitness
    let domain_fitness = fitness_eval.fitness(network_builder.network());

    Fitness {
        domain_fitness: domain_fitness,
        behavioral_diversity: 0.0, // will be calculated in `population_metric`
        connection_cost: connection_cost,
        behavior: behavior,
        age_diversity: 0.0, // will be calculated in `population_metric`
        saturation: sat.sum(),
        complexity: genome.complexity(),
    }
}

fn transformation_from_state(state: &State) -> Transformation {
    Transformation {
        rotate_x: state.rotate_substrate_x,
        rotate_y: state.rotate_substrate_y,
        rotate_z: state.rotate_substrate_z,
        scale_x: state.scale_substrate_x,
        scale_y: state.scale_substrate_y,
        scale_z: state.scale_substrate_z
    }
}

fn main() {
    let mut support = Support::init();

    let mut rng = rand::thread_rng();
    let graph_file = env::args().nth(1).unwrap();
    println!("graph: {}", graph_file);

    let mut domain_fitness_eval = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&graph_file),
        edge_score: false,
        iters: 50,
        eps: 0.01,
    };

    // XXX
    let node_count = domain_fitness_eval.target_graph_node_count();
    println!("{:?}", node_count);

    let substrate_config = substrate_configuration::substrate_configuration(&node_count);

    let mut evo_config = EvoConfig {
        mu: 100,
        lambda: 100,
        k: 2,
        objectives: vec![0, 1, 2, 3, 4, 5],
    };

    let mut selection = SelectNSGPMod { objective_eps: 0.01 };

    let weight_perturbance_sigma = 0.1;
    let link_weight_range = 1.0;
    let mut reproduction = Reproduction {
        mating_method_weights: MatingMethodWeights {
            mutate_add_node: 2,
            mutate_drop_node: 1,
            mutate_modify_node: 0,
            mutate_connect: 2,
            mutate_disconnect: 2,
            mutate_symmetric_join: 2,
            mutate_symmetric_fork: 2,
            mutate_symmetric_connect: 1,
            mutate_weights: 100,
            crossover_weights: 0,
        },
        activation_functions: vec![
                GeometricActivationFunction::Linear,
                GeometricActivationFunction::LinearClipped,
                //GeometricActivationFunction::Gaussian,
                GeometricActivationFunction::BipolarGaussian,
                GeometricActivationFunction::BipolarSigmoid,
                GeometricActivationFunction::Sine,
                GeometricActivationFunction::Absolute,
            ],
        mutate_element_prob: Prob::new(0.05),
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian {
            sigma: weight_perturbance_sigma,
        },
        link_weight_range: WeightRange::bipolar(link_weight_range),
        link_weight_creation_sigma: 0.1,

        mutate_add_node_random_link_weight: true,
        mutate_drop_node_tournament_k: 2,
        mutate_modify_node_tournament_k: 2,
        mate_retries: 100,
    };

    let random_genome_creator = RandomGenomeCreator {
        link_weight_range: WeightRange::bipolar(link_weight_range),

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

    let mut expression = Expression { link_expression_range: (0.1, 0.5) };


    // create `generation 0`
    let mut parents = {
        let mut initial = UnratedPopulation::new();
        for _ in 0..evo_config.mu {
            initial.push(random_genome_creator.create::<_, Position3d>(0, &mut rng));
        }
        let mut rated = initial.rate_in_parallel(&|ind| {
                                                     fitness(ind,
                                                             &expression,
                                                             &substrate_config,
                                                             &domain_fitness_eval)
                                                 },
                                                 INFINITY);

        PopulationFitness.apply(0, &mut rated);

        rated.select(evo_config.mu, &evo_config.objectives, &selection, &mut rng)
    };


    let mut best_individual_i = 0;
    let mut best_fitness = parents.individuals()[best_individual_i].fitness().domain_fitness;
    for (i, ind) in parents.individuals().iter().enumerate() {
        let fitness = ind.fitness().domain_fitness;
        if fitness > best_fitness {
            best_fitness = fitness;
            best_individual_i = i;
        }
    }

    let mut state = State {
        running: false,
        recalc_fitness: false,
        // recalc_substrate
        iteration: 0,
        best_fitness: best_fitness,
        mu: evo_config.mu as i32,
        lambda: evo_config.lambda as i32,
        k: evo_config.k as i32,

        stop_when_fitness_above: 0.999,
        enable_stop: true,

        mutate_add_node: reproduction.mating_method_weights.mutate_add_node as i32,
        mutate_drop_node: reproduction.mating_method_weights.mutate_drop_node as i32,
        mutate_modify_node: reproduction.mating_method_weights.mutate_modify_node as i32,
        mutate_connect: reproduction.mating_method_weights.mutate_connect as i32,
        mutate_disconnect: reproduction.mating_method_weights.mutate_disconnect as i32,

        mutate_symmetric_join: reproduction.mating_method_weights.mutate_symmetric_join as i32,
        mutate_symmetric_fork: reproduction.mating_method_weights.mutate_symmetric_fork as i32,
        mutate_symmetric_connect: reproduction.mating_method_weights
            .mutate_symmetric_connect as i32,

        mutate_weights: reproduction.mating_method_weights.mutate_weights as i32,

        best_fitness_history: vec![(0, best_fitness)],

        nm_edge_score: domain_fitness_eval.edge_score,
        nm_iters: domain_fitness_eval.iters as i32,
        nm_eps: domain_fitness_eval.eps,

        mutate_element_prob: reproduction.mutate_element_prob.get(),
        nsgp_objective_eps: selection.objective_eps as f32,
        weight_perturbance_sigma: weight_perturbance_sigma as f32,
        link_weight_range: link_weight_range as f32,
        link_weight_creation_sigma: reproduction.link_weight_creation_sigma as f32,

        rotate_substrate_x: 45.0,
        rotate_substrate_y: 0.0,
        rotate_substrate_z: 0.0,
        scale_substrate_x: 0.5,
        scale_substrate_y: 0.5,
        scale_substrate_z: 0.5,

        link_expression_min: expression.link_expression_range.0 as f32,
        link_expression_max: expression.link_expression_range.1 as f32,

        objectives_use_behavioral: true,
        objectives_use_cct: true,
        objectives_use_age: true,
        objectives_use_saturation: true,
        objectives_use_complexity: true,

        action: Action::None,
        view: ViewMode::BestDetailed,

        global_mutation_rate: 0.0,
        global_element_mutation: 0.0,

        auto_reset: 250,
        auto_reset_enable: false,
        auto_reset_counter: 0,
    };

    let program_substrate: glium::Program =
        glium::Program::from_source(&support.display,
                                    shaders::VERTEX_SHADER_SUBSTRATE,
                                    shaders::FRAGMENT_SHADER_SUBSTRATE,
                                    None)
            .unwrap();
    let program_vertex: glium::Program =
        glium::Program::from_source(&support.display,
                                    shaders::VERTEX_SHADER_VERTEX,
                                    shaders::FRAGMENT_SHADER_VERTEX,
                                    None)
            .unwrap();

    loop {
        {
            support.render(CLEAR_COLOR, |ui, display, target| {
                const N: usize = 4;
                let (width, height) = target.get_dimensions();
                match state.view { 
                    ViewMode::BestDetailed => {
                        let best_ind = &parents.individuals()[best_individual_i];

                        let (substrate_width, substrate_height) = (400, 400);

                        render_graph(display,
                                     target,
                                     best_ind.genome(),
                                     &expression,
                                     &program_substrate,
                                     &transformation_from_state(&state),
                                     &substrate_config,
                                     glium::Rect {
                                         left: 0,
                                         bottom: 0,
                                         width: substrate_width,
                                         height: substrate_height,
                                     },
                                     2.0,
                                     5.0);


                        render_cppn(display,
                                    target,
                                    best_ind.genome(),
                                    &program_vertex,
                                    glium::Rect {
                                        left: substrate_width,
                                        bottom: 0,
                                        width: width - substrate_width,
                                        height: height,
                                    });
                    }
                    ViewMode::Overview => {
                        let indiv = parents.individuals();
                        let mut indices: Vec<_> = (0..indiv.len()).collect();
                        indices.sort_by(|&i, &j| {
                            match indiv[i]
                                .fitness()
                                .domain_fitness
                                .partial_cmp(&indiv[j].fitness().domain_fitness)
                                .unwrap()
                                .reverse() {
                                Ordering::Equal => {
                                    indiv[i]
                                        .fitness()
                                        .behavioral_diversity
                                        .partial_cmp(&indiv[j].fitness().behavioral_diversity)
                                        .unwrap()
                                        .reverse()
                                }
                                a => a,
                            }
                        });

                        {
                            let mut i = 0;
                            'outer1: for y in 0..N {
                                for x in 0..(2 * N) {
                                    if i >= indiv.len() {
                                        break 'outer1;
                                    }
                                    let rect = glium::Rect {
                                        left: (x as u32) * width / (2 * N as u32),
                                        bottom: (y as u32) * height / (2 * N as u32),
                                        width: width / (2 * N as u32),
                                        height: height / (2 * N as u32),
                                    };
                                    let genome = indiv[indices[i]].genome();
                                    render_cppn(display, target, genome, &program_vertex, rect);
                                    i += 1;
                                }
                            }
                        }

                        {
                            let mut i = 0;
                            'outer2: for y in N..(2 * N) {
                                for x in 0..(2 * N) {
                                    if i >= indiv.len() {
                                        break 'outer2;
                                    }
                                    let rect = glium::Rect {
                                        left: (x as u32) * width / (2 * N as u32),
                                        bottom: (y as u32) * height / (2 * N as u32),
                                        width: width / (2 * N as u32),
                                        height: height / (2 * N as u32),
                                    };
                                    let genome = indiv[indices[i]].genome();
                                    render_graph(display,
                                                 target,
                                                 genome,
                                                 &expression,
                                                 &program_substrate,
                                                 &transformation_from_state(&state),
                                                 &substrate_config,
                                                 rect,
                                                 1.0,
                                                 2.5);
                                    i += 1;
                                }
                            }
                        }

                    }

                    ViewMode::CppnOverview | ViewMode::GraphOverview => {
                        let indiv = parents.individuals();
                        let mut indices: Vec<_> = (0..indiv.len()).collect();
                        indices.sort_by(|&i, &j| {
                            match indiv[i]
                                .fitness()
                                .domain_fitness
                                .partial_cmp(&indiv[j].fitness().domain_fitness)
                                .unwrap()
                                .reverse() {
                                Ordering::Equal => {
                                    indiv[i]
                                        .fitness()
                                        .behavioral_diversity
                                        .partial_cmp(&indiv[j].fitness().behavioral_diversity)
                                        .unwrap()
                                        .reverse()
                                }
                                a => a,
                            }
                        });
                        let mut i = 0;
                        'outer: for y in 0..(2 * N) {
                            for x in 0..(2 * N) {
                                if i >= indiv.len() {
                                    break 'outer;
                                }
                                let rect = glium::Rect {
                                    left: (x as u32) * width / (2 * N as u32),
                                    bottom: (y as u32) * height / (2 * N as u32),
                                    width: width / (2 * N as u32),
                                    height: height / (2 * N as u32),
                                };
                                let genome = indiv[indices[i]].genome();
                                if let ViewMode::CppnOverview = state.view {
                                    render_cppn(display, target, genome, &program_vertex, rect);
                                } else {
                                    render_graph(display,
                                                 target,
                                                 genome,
                                                 &expression,
                                                 &program_substrate,
                                                 &transformation_from_state(&state),
                                                 &substrate_config,
                                                 rect,
                                                 1.0,
                                                 2.5);
                                }
                                i += 1;
                            }
                        }
                    }
                }
                gui(ui, &mut state, &parents);
            });

            let active = support.update_events();
            if !active {
                break;
            }

            if state.running && state.auto_reset_enable && state.action == Action::None {
                if state.iteration > state.auto_reset as usize {
                    println!("Autoreset at {}", state.iteration);
                    state.action = Action::ResetNet;
                    state.auto_reset_counter += 1;
                }
            }

            match state.action {
                Action::ExportBest => {
                    println!("Export best");
                    println!("Iteration: {}", state.iteration);

                    let best = &parents.individuals()[best_individual_i];

                    let basefilename = format!("best.{}.{}",
                                               state.iteration,
                                               (best.fitness().domain_fitness * 1000.0) as usize);

                    println!("filename: {}", basefilename);

                    // Write GML
                    {
                        let mut file = File::create(&format!("{}.gml", basefilename)).unwrap();
                        let mut network_builder = GMLNetworkBuilder::new();
                        network_builder.set_writer(&mut file);
                        network_builder.begin();
                        let (_behavior, _connection_cost, _) = expression.express(best.genome(),
                            &mut network_builder,
                            &substrate_config);
                        network_builder.end();
                    }
                    // Write DOT
                    {
                        let mut file = File::create(&format!("{}.dot", basefilename)).unwrap();
                        let mut network_builder = DotNetworkBuilder::new();
                        network_builder.set_writer(&mut file);
                        network_builder.begin();
                        let (_behavior, _connection_cost, _) = expression.express(best.genome(),
                            &mut network_builder,
                            &substrate_config);
                        network_builder.end();
                    }
                    // Write CPPN
                    {
                        let mut file = File::create(&format!("{}.cppn.dot", basefilename)).unwrap();
                        let network = best.genome().network();

                        writeln!(&mut file,
                                 "digraph {{
graph [
  layout=neato,
  rankdir = \"TB\",
  \
                                  overlap=false,
  compound = true,
  nodesep = 1,
  ranksep = \
                                  2.0,
  splines = \"polyline\",
];
node [fontname = Helvetica];
")
                            .unwrap();

                        network.each_node_with_index(|node, node_idx| {
                            let s = match node.node_type().kind {
                                CppnNodeKind::Input => {
                                    let label = match node_idx.index() {
                                        0 => "x1",
                                        1 => "y1",
                                        2 => "z1",
                                        3 => "x2",
                                        4 => "y2",
                                        5 => "z2",
                                        _ => "X",
                                    };

                                    // XXX label
                                    format!("shape=egg,label={},rank=min,style=filled,color=grey",
                                            label)
                                }
                                CppnNodeKind::Bias => {
                                    assert!(node_idx.index() == 6);
                                    format!("shape=egg,label=1.0,rank=min,style=filled,color=grey")
                                }
                                CppnNodeKind::Output => {
                                    let label = match node_idx.index() {
                                        7 => "t",
                                        8 => "ex",
                                        9 => "w",
                                        10 => "r", 
                                        _ => panic!(),
                                    };
                                    format!("shape=doublecircle,label={},rank=max,style=filled,\
                                             fillcolor=yellow,color=grey",
                                            label)
                                }
                                CppnNodeKind::Hidden => {
                                    format!("shape=box,label={}",
                                            node.node_type().activation_function.name())
                                }
                            };
                            writeln!(&mut file, "{} [{}];", node_idx.index(), s).unwrap();
                        });
                        network.each_link_ref(|link_ref| {
                            let w = link_ref.link().weight().0;
                            let color = if w >= 0.0 { "black" } else { "red" };
                            writeln!(&mut file,
                                     "{} -> {} [color={}];",
                                     link_ref.link().source_node_index().index(),
                                     link_ref.link().target_node_index().index(),
                                     color)
                                .unwrap();
                        });

                        writeln!(&mut file, "}}").unwrap();
                    }

                }
                Action::ResetNet => {
                    parents = {
                        let mut initial = UnratedPopulation::new();
                        for _ in 0..state.mu {
                            initial.push(random_genome_creator.create::<_, Position3d>(0, &mut rng));
                        }
                        let mut rated = initial.rate_in_parallel(&|ind| {
                                                                     fitness(ind,
                                                                             &expression,
                                                                             &substrate_config,
                                                                             &domain_fitness_eval)
                                                                 },
                                                                 INFINITY);

                        PopulationFitness.apply(0, &mut rated);

                        rated.select(state.mu as usize,
                                     &evo_config.objectives,
                                     &selection,
                                     &mut rng)
                    };

                    best_individual_i = 0;
                    best_fitness =
                        parents.individuals()[best_individual_i].fitness().domain_fitness;
                    for (i, ind) in parents.individuals().iter().enumerate() {
                        let fitness = ind.fitness().domain_fitness;
                        if fitness > best_fitness {
                            best_fitness = fitness;
                            best_individual_i = i;
                        }
                    }
                    state.best_fitness = best_fitness;
                    state.best_fitness_history.clear();
                    state.best_fitness_history.push((state.iteration, state.best_fitness));
                    state.iteration = 0;
                }
                _ => {}
            }
            state.action = Action::None;

            evo_config.mu = state.mu as usize;
            evo_config.lambda = state.lambda as usize;
            evo_config.k = state.k as usize;
            reproduction.mating_method_weights.mutate_add_node = state.mutate_add_node as u32;
            reproduction.mating_method_weights.mutate_drop_node = state.mutate_drop_node as u32;
            reproduction.mating_method_weights.mutate_modify_node = state.mutate_modify_node as u32;
            reproduction.mating_method_weights.mutate_connect = state.mutate_connect as u32;
            reproduction.mating_method_weights.mutate_disconnect = state.mutate_disconnect as u32;

            reproduction.mating_method_weights.mutate_symmetric_join =
                state.mutate_symmetric_join as u32;
            reproduction.mating_method_weights.mutate_symmetric_fork =
                state.mutate_symmetric_fork as u32;
            reproduction.mating_method_weights.mutate_symmetric_connect =
                state.mutate_symmetric_connect as u32;

            reproduction.mating_method_weights.mutate_weights = state.mutate_weights as u32;
            domain_fitness_eval.edge_score = state.nm_edge_score;
            domain_fitness_eval.iters = state.nm_iters as usize;
            domain_fitness_eval.eps = state.nm_eps;


            reproduction.mutate_element_prob = Prob::new(state.mutate_element_prob);
            selection.objective_eps = state.nsgp_objective_eps as f64;
            reproduction.weight_perturbance = WeightPerturbanceMethod::JiggleGaussian {
                sigma: state.weight_perturbance_sigma as f64,
            };
            reproduction.link_weight_range = WeightRange::bipolar(state.link_weight_range as f64);
            reproduction.link_weight_creation_sigma = state.link_weight_creation_sigma as f64;

            expression.link_expression_range = (state.link_expression_min as f64,
                                                state.link_expression_max as f64);

            let mut new_objectives = Vec::new();
            new_objectives.push(0);
            if state.objectives_use_behavioral {
                new_objectives.push(1);
            }
            if state.objectives_use_cct {
                new_objectives.push(2);
            }
            if state.objectives_use_age {
                new_objectives.push(3);
            }
            if state.objectives_use_saturation {
                new_objectives.push(4);
            }
            if state.objectives_use_complexity {
                new_objectives.push(5);
            }

            if evo_config.objectives != new_objectives {
                evo_config.objectives = new_objectives;
            }

            if state.recalc_fitness {
                // XXX: Action::RecalcFitness
                state.recalc_fitness = false;
                let offspring = parents.into_unrated();
                let mut next_gen = offspring.rate_in_parallel(&|ind| {
                                                                  fitness(ind,
                                                                          &expression,
                                                                          &substrate_config,
                                                                          &domain_fitness_eval)
                                                              },
                                                              INFINITY);

                PopulationFitness.apply(state.iteration, &mut next_gen);
                parents =
                    next_gen.select(evo_config.mu, &evo_config.objectives, &selection, &mut rng);

                best_individual_i = 0;
                best_fitness = parents.individuals()[best_individual_i].fitness().domain_fitness;
                for (i, ind) in parents.individuals().iter().enumerate() {
                    let fitness = ind.fitness().domain_fitness;
                    if fitness > best_fitness {
                        best_fitness = fitness;
                        best_individual_i = i;
                    }
                }

                state.best_fitness = best_fitness;
                state.best_fitness_history.push((state.iteration, state.best_fitness));
            }
        }

        if state.enable_stop && state.best_fitness >= state.stop_when_fitness_above as f64 {
            state.running = false;
        }

        if state.running {
            let time_before = time::precise_time_ns();

            // create next generation
            state.iteration += 1;
            let offspring =
                parents.reproduce(&mut rng,
                                  evo_config.lambda,
                                  evo_config.k,
                                  &|rng, p1, p2| reproduction.mate(rng, p1, p2, state.iteration));
            let mut next_gen = if state.global_mutation_rate > 0.0 {
                // mutate all individuals of the whole population.
                // XXX: Optimize
                let old = parents.into_unrated().merge(offspring);
                let mut new_unrated = UnratedPopulation::new();
                let prob = Prob::new(state.global_mutation_rate);
                for ind in old.as_vec().into_iter() {
                    // mutate each in
                    let mut genome = ind.into_genome();
                    if prob.flip(&mut rng) {
                        // mutate that genome
                        genome.mutate_weights(Prob::new(state.global_element_mutation),
                                              &reproduction.weight_perturbance,
                                              &reproduction.link_weight_range,
                                              &mut rng);
                    }
                    new_unrated.push(genome);
                }
                new_unrated.rate_in_parallel(&|ind| {
                                                 fitness(ind,
                                                         &expression,
                                                         &substrate_config,
                                                         &domain_fitness_eval)
                                             },
                                             INFINITY)
            } else {
                // no global mutation.
                let rated_offspring = offspring.rate_in_parallel(&|ind| {
                                                                     fitness(ind,
                                                                             &expression,
                                                                             &substrate_config,
                                                                             &domain_fitness_eval)
                                                                 },
                                                                 INFINITY);
                parents.merge(rated_offspring)
            };
            PopulationFitness.apply(state.iteration, &mut next_gen);
            parents = next_gen.select(evo_config.mu, &evo_config.objectives, &selection, &mut rng);


            best_individual_i = 0;
            best_fitness = parents.individuals()[best_individual_i].fitness().domain_fitness;
            for (i, ind) in parents.individuals().iter().enumerate() {
                let fitness = ind.fitness().domain_fitness;
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_individual_i = i;
                }
            }

            state.best_fitness = best_fitness;
            state.best_fitness_history.push((state.iteration, state.best_fitness));

            let time_after = time::precise_time_ns();
            assert!(time_after > time_before);
            let total_ns = time_after - time_before;
            println!("{}\t{}", state.iteration, total_ns);
        }
    }
}
