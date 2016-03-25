extern crate hypernsga;
extern crate nsga2;
extern crate rand;
extern crate criterion_stats;

#[macro_use]
extern crate glium;
#[macro_use]
extern crate imgui;
extern crate time;
extern crate imgui_sys;
extern crate libc;
extern crate graph_layout;

use hypernsga::graph;
use hypernsga::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use hypernsga::network_builder::NetworkBuilder;
use hypernsga::cppn::{Cppn, GeometricActivationFunction, RandomGenomeCreator, Reproduction,
Expression, G, PopulationFitness};
use hypernsga::fitness::{Fitness, DomainFitness};
use hypernsga::mating::MatingMethodWeights;
use hypernsga::prob::Prob;
use hypernsga::weight::{WeightPerturbanceMethod, WeightRange};
use hypernsga::substrate::{Node, Substrate, SubstrateConfiguration, Position, Position3d, Position2d,
NodeConnectivity};
use hypernsga::distribute::DistributeInterval;
use nsga2::selection::SelectNSGP;
use nsga2::population::{UnratedPopulation, RatedPopulation, RankedPopulation};
use std::f64::INFINITY;
use criterion_stats::univariate::Sample;
use std::env;
use std::mem;
use rand::Rng;

use imgui::*;
use self::support::Support;
use imgui_sys::igPlotLines2;
use libc::*;
use glium::Surface;
use glium::index::PrimitiveType;

mod support;

const CLEAR_COLOR: (f32, f32, f32, f32) = (1.0, 1.0, 1.0, 1.0);

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

#[derive(Copy, Clone)]
struct Point {
    position: [f32; 2],
}

implement_vertex!(Vertex, position, color);
implement_vertex!(Point, position);


pub struct VizNetworkBuilder {
    point_list: Vec<Point>,
    link_index_list: Vec<u32>,
}

impl NetworkBuilder for VizNetworkBuilder {
    type POS = Position2d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        VizNetworkBuilder {
            point_list: Vec::new(),
            link_index_list: Vec::new(),
        }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, _param: f64) {
        assert!(node.index == self.point_list.len());
        self.point_list.push(Point{position: [node.position.x() as f32, node.position.y() as f32]});
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let w = weight1.abs();
        debug_assert!(w <= 1.0);

        self.link_index_list.push(source_node.index as u32);
        self.link_index_list.push(target_node.index as u32);
        //let _ = self.builder.add_edge(source_node.index, target_node.index, Closed01::new(w as f32)); 
    }

    fn network(self) -> Self::Output {
        ()
    }
}

#[derive(Debug)]
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

    mutate_element_prob: f32,
    nsgp_objective_eps: f32,
    weight_perturbance_sigma: f32,
    link_weight_range: f32,
    link_weight_creation_sigma: f32,

    best_fitness_history: Vec<(usize, f64)>,
    //crossover_weights: 0,


    nm_edge_score: bool,
    nm_iters: i32,
    nm_eps: f32,
}

struct EvoConfig {
    mu: usize,
    lambda: usize,
    k: usize,
    num_objectives: usize,
}

extern "C" fn values_getter(data: *mut c_void, idx: c_int) -> c_float {
    unsafe {
        let state: &mut State = mem::transmute(data);
        state.best_fitness_history.get(idx as usize).map(|e| e.1).unwrap() as c_float
    }
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
                ui.separator();

                let num_points = state.best_fitness_history.len();
                unsafe {
                    igPlotLines2(im_str!("performance").as_ptr(),
                    values_getter,
                    (state as *mut State) as *mut c_void,
                    num_points as c_int,
                    0 as c_int,
                    im_str!("overlay text").as_ptr(),
                    0.0 as c_float,
                    1.0 as c_float,
                    ImVec2::new(400.0, 50.0)
                    );
                }
            }
            if ui.collapsing_header(im_str!("Population Settings")).build() {
                ui.slider_i32(im_str!("Population Size"), &mut state.mu, state.k, 1000).build();
                ui.slider_i32(im_str!("Offspring Size"), &mut state.lambda, 1, 1000).build();
            }

            if ui.collapsing_header(im_str!("Selection")).build() {
                ui.slider_i32(im_str!("Tournament Size"), &mut state.k, 1, state.mu).build();
                ui.slider_f32(im_str!("NSGP Objective Epsilon"), &mut state.nsgp_objective_eps, 0.0, 1.0).build();
            }

            if ui.collapsing_header(im_str!("CPPN")).build() {
                ui.slider_f32(im_str!("Link Weight Range (bipolar)"), &mut state.link_weight_range, 0.1, 5.0).build();
                ui.slider_f32(im_str!("Link Weight Creation Sigma"), &mut state.link_weight_creation_sigma, 0.01, 1.0).build();
                ui.slider_f32(im_str!("Weight Perturbance Sigma"), &mut state.weight_perturbance_sigma, 0.0, 1.0).build();
            }

            if ui.collapsing_header(im_str!("Neighbor Matching")).build() {
                ui.checkbox(im_str!("Edge Weight Scoring"), &mut state.nm_edge_score);
                ui.slider_i32(im_str!("Iterations"), &mut state.nm_iters, 1, 1000).build();
                ui.slider_f32(im_str!("Eps"), &mut state.nm_eps, 0.0, 1.0).build();
            }

            if ui.collapsing_header(im_str!("Mutation")).build() {
                ui.slider_f32(im_str!("Mutation Rate"), &mut state.mutate_element_prob, 0.0, 1.0).build();
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



fn fitness<P>(genome: &G,
              expression: &Expression,
              substrate_config: &SubstrateConfiguration<P, Neuron>,
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

    let mut domain_fitness_eval = GraphSimilarity {
        target_graph: graph::load_graph_normalized(&graph_file),
        edge_score: true,
        iters: 50,
        eps: 0.01,
    };

    // XXX
    let mut substrate: Substrate<Position2d, Neuron> = Substrate::new();
    let node_count = domain_fitness_eval.target_graph_node_count();

    println!("{:?}", node_count);

    let min = -1.0;
    let max = 1.0;

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

    let mut selection = SelectNSGP { objective_eps: 0.1 };

    let weight_perturbance_sigma = 0.1;
    let link_weight_range = 1.0;
    let mut reproduction = Reproduction {
        mating_method_weights: MatingMethodWeights {
            mutate_add_node: 1,
            mutate_drop_node: 1,
            mutate_modify_node: 1,
            mutate_connect: 20,
            mutate_disconnect: 20,
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
        weight_perturbance: WeightPerturbanceMethod::JiggleGaussian { sigma: weight_perturbance_sigma },
        link_weight_range: WeightRange::bipolar(link_weight_range),
        link_weight_creation_sigma: 0.1,

        mutate_add_node_random_link_weight: true,
        mutate_drop_node_tournament_k: 10,
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

    let expression = Expression { link_expression_threshold: 0.1 };

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
        running: true,
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

        best_fitness_history: vec![(0, best_fitness)],

        nm_edge_score: domain_fitness_eval.edge_score,
        nm_iters: domain_fitness_eval.iters as i32,
        nm_eps: domain_fitness_eval.eps,

        mutate_element_prob: reproduction.mutate_element_prob.get(),
        nsgp_objective_eps: selection.objective_eps as f32,
        weight_perturbance_sigma: weight_perturbance_sigma as f32,
        link_weight_range: link_weight_range as f32,
        link_weight_creation_sigma: reproduction.link_weight_creation_sigma as f32,
    };

    let mut program: Option<glium::Program> = None;
    let mut program_vertex: Option<glium::Program> = None;

    loop {
        {
            support.render(CLEAR_COLOR, |display, imgui, renderer, target, delta_f| {
                let best_ind = &parents.individuals()[best_individual_i];

                let mut network_builder = VizNetworkBuilder::new();
                let (_, _) = expression.express(&best_ind.genome(),
                &mut network_builder,
                &substrate_config);

                let vertex_buffer = {
                    glium::VertexBuffer::new(display, &network_builder.point_list).unwrap()
                };

                let point_index_buffer = glium::index::NoIndices(PrimitiveType::Points);
                let line_index_buffer  = glium::IndexBuffer::new(display, PrimitiveType::LinesList,
                                                                 &network_builder.link_index_list).unwrap();


                // Layout the CPPN
                let cppn = Cppn::new(best_ind.genome().network());
                let layers = cppn.group_layers();
                let mut dy = DistributeInterval::new(layers.len(), -1.0, 1.0);

                let mut node_positions: Vec<_> = best_ind.genome().network().nodes().iter().map(|node| {
                    graph_layout::P2d(0.0, 0.0)
                }).collect();

                for layer in layers {
                    let y = dy.next().unwrap();
                    let mut dx = DistributeInterval::new(layer.len(), -1.0, 1.0);
                    for nodeidx in layer {
                        let x = dx.next().unwrap();
                        node_positions[nodeidx] = graph_layout::P2d(x as f32, y as f32);
                    }
                }

                let mut cppn_links = Vec::new();
                best_ind.genome().network().each_link_ref(|link_ref| {
                    let src = link_ref.link().source_node_index().index();
                    let dst = link_ref.link().target_node_index().index();
                    cppn_links.push(src as u32);
                    cppn_links.push(dst as u32);
                });

                let cppn_vertices: Vec<_> = node_positions.iter().map(|p2d| Vertex{position: [p2d.0, p2d.1], color: [0.0, 1.0, 0.0]}
                                                                     ).collect();



                let vertex_buffer_cppn = {
                    glium::VertexBuffer::new(display, &cppn_vertices).unwrap()
                };

                let cppn_index_buffer = glium::IndexBuffer::new(display, PrimitiveType::LinesList, &cppn_links).unwrap();

                if program.is_none() {
                    program = Some(program!(display,
                                            140 => {
                                                vertex: "
                    #version 140
                    uniform mat4 matrix;
                    in vec2 position;
                    void main() {
                        gl_Position = matrix * vec4(position, 0.0, 1.0);
                    }
                ",

                fragment: "
                    #version 140
                    out vec4 color;
                    void main() {
                        color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                "
                                            },
                                            ).unwrap());
                }

                if program_vertex.is_none() {
                    program_vertex = Some(program!(display,
                                                   140 => {
                                                       vertex: "
                    #version 140
                    uniform mat4 matrix;
                    in vec2 position;
                    in vec3 color;
                    out vec3 fl_color;
                    void main() {
                        gl_Position = matrix * vec4(position, 0.0, 1.0);
                        fl_color = color;
                    }
                ",

                fragment: "
                    #version 140
                    in vec3 fl_color;
                    out vec4 color;
                    void main() {
                        color = vec4(fl_color, 1.0);
                    }
                "
                                                   },
                                                   ).unwrap());
                }

                let uniforms = uniform! {
                    matrix: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0f32]
                    ]
                };

                let draw_parameters = glium::draw_parameters::DrawParameters {
                    line_width: Some(1.0),
                    point_size: Some(10.0),
                    .. Default::default()
                };
                target.draw(&vertex_buffer, &point_index_buffer, program.as_ref().unwrap(), &uniforms, &draw_parameters).unwrap();
                target.draw(&vertex_buffer, &line_index_buffer, program.as_ref().unwrap(), &uniforms, &draw_parameters).unwrap();
                target.draw(&vertex_buffer_cppn, &glium::index::NoIndices(PrimitiveType::Points), program_vertex.as_ref().unwrap(), &uniforms, &draw_parameters).unwrap();
                target.draw(&vertex_buffer_cppn, &cppn_index_buffer, program_vertex.as_ref().unwrap(), &uniforms, &draw_parameters).unwrap();

                let (width, height) = target.get_dimensions();
                let ui = imgui.frame(width, height, delta_f);
                gui(&ui, &mut state);
                renderer.render(target, ui).unwrap();
            }
            );

            evo_config.mu = state.mu as usize;
            evo_config.lambda = state.lambda as usize;
            evo_config.k = state.k as usize;
            reproduction.mating_method_weights.mutate_add_node = state.mutate_add_node as u32;
            reproduction.mating_method_weights.mutate_drop_node = state.mutate_drop_node as u32;
            reproduction.mating_method_weights.mutate_modify_node = state.mutate_modify_node as u32;
            reproduction.mating_method_weights.mutate_connect = state.mutate_connect as u32;
            reproduction.mating_method_weights.mutate_disconnect = state.mutate_disconnect as u32;
            reproduction.mating_method_weights.mutate_weights = state.mutate_weights as u32;
            domain_fitness_eval.edge_score = state.nm_edge_score;
            domain_fitness_eval.iters = state.nm_iters as usize;
            domain_fitness_eval.eps = state.nm_eps;


            reproduction.mutate_element_prob = Prob::new(state.mutate_element_prob);
            selection.objective_eps = state.nsgp_objective_eps as f64;
            reproduction.weight_perturbance = WeightPerturbanceMethod::JiggleGaussian { 
                sigma: state.weight_perturbance_sigma as f64};
            reproduction.link_weight_range = WeightRange::bipolar(state.link_weight_range as f64);
            reproduction.link_weight_creation_sigma = state.link_weight_creation_sigma as f64;

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
}
