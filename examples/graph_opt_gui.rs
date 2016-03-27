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
use hypernsga::substrate::{Node, NodeSet, Substrate, SubstrateConfiguration, Position, Position3d, Position2d};
use hypernsga::placement;
use hypernsga::distribute::DistributeInterval;
use nsga2::selection::{SelectNSGP,SelectNSGPMod};
use nsga2::population::{UnratedPopulation, RatedPopulation, RankedPopulation};
use nsga2::multi_objective::MultiObjective;
use std::f64::INFINITY;
use criterion_stats::univariate::Sample;
use std::env;
use std::mem;
use rand::Rng;

use imgui::*;
use self::support::Support;
use imgui_sys::{igPlotLines2, igCombo2};
use libc::*;
use glium::Surface;
use glium::index::PrimitiveType;
use std::io::Write;
use std::fs::File;
use glium::backend::glutin_backend::GlutinFacade;
use std::cmp::Ordering;

mod support;

const CLEAR_COLOR: (f32, f32, f32, f32) = (1.0, 1.0, 1.0, 1.0);

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

implement_vertex!(Vertex, position, color);

pub struct VizNetworkBuilder {
    point_list: Vec<Vertex>,
    link_index_list: Vec<u32>,
}

impl NetworkBuilder for VizNetworkBuilder {
    type POS = Position3d;
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
        let color = match node.node_info {
            Neuron::Input => [0.0, 1.0, 0.0, 1.0],
            Neuron::Hidden => [0.0, 0.0, 0.0, 1.0],
            Neuron::Output => [1.0, 0.0, 0.0, 1.0],
        };
        self.point_list.push(Vertex {
            position: [node.position.x() as f32,
            node.position.y() as f32,
            node.position.z() as f32],
            color: color,
        });
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
    }

    fn network(self) -> Self::Output {
        ()
    }
}

pub struct GMLNetworkBuilder<'a, W: Write+'a> {
    wr: Option<&'a mut W>
}

impl<'a, W:Write> GMLNetworkBuilder<'a, W> {
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

impl<'a, W:Write> NetworkBuilder for GMLNetworkBuilder<'a, W> {
    type POS = Position3d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        GMLNetworkBuilder {
            wr: None
        }
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
        writeln!(wr, "  edge [source {} target {} weight {:.1}]", source_node.index, target_node.index, w).unwrap();
    }

    fn network(self) -> Self::Output {
        ()
    }
}

#[derive(Debug)]
enum ViewMode {
    BestDetailed,
    CppnOverview,
    GraphOverview,
    Overview,
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
    mutate_symmetric_join: i32,
    mutate_symmetric_fork: i32,
    mutate_symmetric_connect: i32,
    mutate_weights: i32,

    mutate_element_prob: f32,
    nsgp_objective_eps: f32,
    weight_perturbance_sigma: f32,
    link_weight_range: f32,
    link_weight_creation_sigma: f32,

    best_fitness_history: Vec<(usize, f64)>,
    // crossover_weights: 0,
    nm_edge_score: bool,
    nm_iters: i32,
    nm_eps: f32,

    rotate_substrate_x: f32,
    rotate_substrate_y: f32,
    rotate_substrate_z: f32,
    scale_substrate_x: f32,
    scale_substrate_y: f32,
    scale_substrate_z: f32,

    link_expression_min: f32,
    link_expression_max: f32,

    recalc_fitness: bool,

    objectives_use_behavioral: bool,
    objectives_use_cct: bool,
    objectives_use_age: bool,
    objectives_use_saturation: bool,
    objectives_use_complexity: bool,

    action: Action,
    view: ViewMode,

    global_mutation_rate: f32,
    global_element_mutation: f32,
}

struct EvoConfig {
    mu: usize,
    lambda: usize,
    k: usize,
    objectives: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
enum Action {
    None,
    ExportBest,
    ResetNet,
}

extern "C" fn values_getter(data: *mut c_void, idx: c_int) -> c_float {
    unsafe {
        let state: &mut State = mem::transmute(data);
        state.best_fitness_history.get(idx as usize).map(|e| e.1).unwrap() as c_float
    }
}

fn render_graph(display: &GlutinFacade, target: &mut glium::Frame, genome: &G, expression: &Expression, program: &glium::Program, state: &State,
                substrate_config: &SubstrateConfiguration<Position3d, Neuron>, viewport: glium::Rect, line_width: f32, point_size: f32) {
    let mut network_builder = VizNetworkBuilder::new();
    let (_, _, _) = expression.express(&genome,
                                       &mut network_builder,
                                       &substrate_config);

    let vertex_buffer = glium::VertexBuffer::new(display, &network_builder.point_list).unwrap();

    let line_index_buffer  = glium::IndexBuffer::new(display, PrimitiveType::LinesList,
                                                     &network_builder.link_index_list).unwrap();

    let rx = state.rotate_substrate_x.to_radians();
    let ry = state.rotate_substrate_y.to_radians();
    let rz = state.rotate_substrate_z.to_radians();
    let sx = state.scale_substrate_x;
    let sy = state.scale_substrate_y;
    let sz = state.scale_substrate_z;

    let perspective = {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32],
        ]
    };

    let uniforms_substrate = uniform! {
        matrix: [
            [sx*ry.cos()*rz.cos(), -ry.cos()*rz.sin(), ry.sin(), 0.0],
            [rx.cos()*rz.sin() + rx.sin()*ry.sin()*rz.cos(), sy*(rx.cos()*rz.cos() - rx.sin()*ry.sin()*rz.sin()), -rx.sin()*ry.cos(), 0.0],
            [rx.sin()*rz.sin() - rx.cos()*ry.sin()*rz.cos(), rx.sin()*rz.cos() + rx.cos()*ry.sin()*rz.sin(), sz*rx.cos()*ry.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ],
        perspective: perspective
    };

    let draw_parameters_substrate = glium::draw_parameters::DrawParameters {
        line_width: Some(line_width),
        blend: glium::Blend::alpha_blending(),
        smooth: Some(glium::draw_parameters::Smooth::Nicest),
        viewport: Some(viewport),
        .. Default::default()
    };

    // substrate
    target.draw(&vertex_buffer, &line_index_buffer, program, &uniforms_substrate, &draw_parameters_substrate).unwrap();

    let draw_parameters_substrate = glium::draw_parameters::DrawParameters {
        point_size: Some(point_size),
        viewport: Some(viewport),
        .. Default::default()
    };

    let point_index_buffer = glium::index::NoIndices(PrimitiveType::Points);
    target.draw(&vertex_buffer, &point_index_buffer, program, &uniforms_substrate, &draw_parameters_substrate).unwrap();
}

fn render_cppn(display: &GlutinFacade, target: &mut glium::Frame, genome: &G, expression: &Expression, program: &glium::Program, state: &State,
               substrate_config: &SubstrateConfiguration<Position3d, Neuron>, viewport: glium::Rect) {

    // Layout the CPPN
    let cppn = Cppn::new(genome.network());
    let layers = cppn.group_layers();
    let mut dy = DistributeInterval::new(layers.len(), -1.0, 1.0);

    let mut cppn_node_positions: Vec<_> = genome.network().nodes().iter().map(|node| {
        Vertex{position: [0.0, 0.0, 0.0], color: [0.0, 1.0, 0.0, 1.0]}
    }).collect();

    let mut cppn_node_triangles = Vec::new();

    let mut line_vertices = Vec::new();

    for layer in layers {
        let y = dy.next().unwrap();
        let mut dx = DistributeInterval::new(layer.len(), -1.0, 1.0);
        for nodeidx in layer {
            let x = dx.next().unwrap() as f32;
            let y = -y as f32;
            cppn_node_positions[nodeidx].position[0] = x;
            cppn_node_positions[nodeidx].position[1] = y;

            let node = &genome.network().nodes()[nodeidx];
            let w = 0.03;
            let aspect = viewport.width as f32 / viewport.height as f32;
            let h = aspect * w;

            match node.node_type().activation_function {
                GeometricActivationFunction::Linear => {
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y,   0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y+h, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                }
                GeometricActivationFunction::BipolarSigmoid => {
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y,   0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y+h, 0.0], color:   [1.0, 0.0, 1.0, 1.0]});
                }
                GeometricActivationFunction::BipolarGaussian => {
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y,   0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x, y+h, 0.0], color:   [1.0, 0.0, 1.0, 1.0]});
                }
                GeometricActivationFunction::Sine => {
                    cppn_node_triangles.push(Vertex{position: [x, y,   0.0], color:   [1.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y+h, 0.0], color:   [1.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y+h, 0.0], color:   [1.0, 0.0, 1.0, 1.0]});
                }
                _ => {
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y, 0.0], color:   [1.0, 0.0, 0.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y+h, 0.0], color:   [1.0, 0.0, 0.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0),   y, 0.0], color:   [1.0, 0.0, 0.0, 1.0]});

                    /*
                    cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y+h, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y+h, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    cppn_node_triangles.push(Vertex{position: [x+(w/2.0),   y, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    */
                }
            }

        }
    }

    let mut cppn_links = Vec::new();
    genome.network().each_link_ref(|link_ref| {
        let src = link_ref.link().source_node_index().index();
        let dst = link_ref.link().target_node_index().index();
        cppn_links.push(src as u32);
        cppn_links.push(dst as u32);

        let src_x = cppn_node_positions[src].position[0];
        let src_y = cppn_node_positions[src].position[1];
        let dst_x = cppn_node_positions[dst].position[0];
        let dst_y = cppn_node_positions[dst].position[1];

        let weight = link_ref.link().weight().0 as f32;
        assert!(weight.abs() <= 1.0); // XXX
        let wa = (weight.abs()/2.0)+0.5;
        let color =
            if weight >= 0.0 {
                [0.0, 1.0, 0.0, wa]
            } else {
                [1.0, 0.0, 0.0, wa]
        };

        line_vertices.push(Vertex{position: [src_x, src_y, 0.0], color: color});
        line_vertices.push(Vertex{position: [dst_x, dst_y, 0.0], color: color});
    });

    let vertex_buffer_cppn = glium::VertexBuffer::new(display, &cppn_node_positions).unwrap();
    let cppn_index_buffer = glium::IndexBuffer::new(display, PrimitiveType::LinesList, &cppn_links).unwrap();

    let triangle_buffer = glium::VertexBuffer::new(display, &cppn_node_triangles).unwrap();

    let uniforms_cppn = uniform! {
        matrix: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ]
    };

    let draw_parameters2 = glium::draw_parameters::DrawParameters {
        line_width: Some(3.0),
        blend: glium::Blend::alpha_blending(),
        smooth: Some(glium::draw_parameters::Smooth::Nicest),
        viewport: Some(viewport),
        .. Default::default()
    };

    //target.draw(&vertex_buffer_cppn, &glium::index::NoIndices(PrimitiveType::Points), program, &uniforms_cppn, &draw_parameters2).unwrap();
    //target.draw(&vertex_buffer_cppn, &cppn_index_buffer, program, &uniforms_cppn, &draw_parameters2).unwrap();

    let lines_buffer = glium::VertexBuffer::new(display, &line_vertices).unwrap();
    target.draw(&lines_buffer, &glium::index::NoIndices(PrimitiveType::LinesList), program, &uniforms_cppn, &draw_parameters2).unwrap();
    target.draw(&triangle_buffer, &glium::index::NoIndices(PrimitiveType::TrianglesList), program, &uniforms_cppn, &draw_parameters2).unwrap();
}




fn gui<'a>(ui: &Ui<'a>, state: &mut State, population: &RankedPopulation<G, Fitness>) {
    ui.window(im_str!("Evolutionary Graph Optimization"))
        .size((300.0, 100.0), ImGuiSetCond_FirstUseEver)
        .build(|| {
            // if ui.collapsing_header(im_str!("General")).build() {
            ui.text(im_str!("Iteration: {}", state.iteration));
            ui.text(im_str!("Best Fitness: {:.3}", state.best_fitness));
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
            if ui.small_button(im_str!("Recalc Fitness")) {
                state.recalc_fitness = true;
            }
            if ui.small_button(im_str!("Export Best")) {
                state.action = Action::ExportBest;
            }
            if ui.small_button(im_str!("Reset")) {
                state.action = Action::ResetNet;
            }

            let views = im_str!("detailed\0multi cppn\0multi graph\0overview\0");
            let mut current: c_int = match state.view {
                ViewMode::BestDetailed => {
                    0
                }
                ViewMode::CppnOverview => {
                    1
                }
                ViewMode::GraphOverview => {
                    2
                }
                ViewMode::Overview => {
                    3
                }

            };
            unsafe {
                if igCombo2(im_str!("view").as_ptr(), &mut current as *mut c_int, views.as_ptr(), 4) {
                    if current == 0 {
                        state.view = ViewMode::BestDetailed;
                    } else if current == 1 {
                        state.view = ViewMode::CppnOverview;
                    } else if current == 2 {
                        state.view = ViewMode::GraphOverview;
                    } else if current == 3 {
                        state.view = ViewMode::Overview;
                    }


                }
            }

            if ui.collapsing_header(im_str!("Population Metrics")).build() {
                let best_domain = population.individuals().iter().max_by_key(|ind| (ind.fitness().domain_fitness * 1_000_000.0) as usize).unwrap();
                let worst_domain = population.individuals().iter().min_by_key(|ind| (ind.fitness().domain_fitness * 1_000_000.0) as usize).unwrap();
                ui.text(im_str!("Best Domain Fitness: {:.3}", best_domain.fitness().domain_fitness));
                ui.text(im_str!("Age of Best: {}", best_domain.genome().age(state.iteration)));

                ui.text(im_str!("Worst Domain Fitness: {:.3}", worst_domain.fitness().domain_fitness));
                ui.text(im_str!("Age of Worst: {}", worst_domain.genome().age(state.iteration)));
            }

            if ui.collapsing_header(im_str!("History")).build() {
                let num_points = state.best_fitness_history.len();
                unsafe {
                    igPlotLines2(im_str!("performance").as_ptr(),
                    values_getter,
                    (state as *mut State) as *mut c_void,
                    num_points as c_int,
                    0 as c_int,
                    im_str!("Domain Fitness").as_ptr(),
                    0.0 as c_float,
                    1.0 as c_float,
                    ImVec2::new(400.0, 50.0));
                }
            }
            if ui.collapsing_header(im_str!("Objectives")).build() {
                ui.checkbox(im_str!("Behavioral Diversity"), &mut state.objectives_use_behavioral);
                ui.checkbox(im_str!("Connection Cost"), &mut state.objectives_use_cct);
                ui.checkbox(im_str!("Age Diversity"), &mut state.objectives_use_age);
                ui.checkbox(im_str!("Saturation"), &mut state.objectives_use_saturation);
                ui.checkbox(im_str!("Complexity"), &mut state.objectives_use_complexity);
            }
            if ui.collapsing_header(im_str!("Population Settings")).build() {
                ui.slider_i32(im_str!("Population Size"), &mut state.mu, state.k, 1000).build();
                ui.slider_i32(im_str!("Offspring Size"), &mut state.lambda, 1, 1000).build();
            }

            if ui.collapsing_header(im_str!("Selection")).build() {
                ui.slider_i32(im_str!("Tournament Size"), &mut state.k, 1, state.mu).build();
                ui.slider_f32(im_str!("NSGP Objective Epsilon"),
                &mut state.nsgp_objective_eps,
                0.0,
                1.0)
                    .build();
            }

            if ui.collapsing_header(im_str!("View")).build() {
                ui.slider_f32(im_str!("Rotate Substrate x"),
                &mut state.rotate_substrate_x,
                0.0,
                360.0)
                    .build();
                ui.slider_f32(im_str!("Rotate Substrate y"),
                &mut state.rotate_substrate_y,
                0.0,
                360.0)
                    .build();
                ui.slider_f32(im_str!("Rotate Substrate z"),
                &mut state.rotate_substrate_z,
                0.0,
                360.0)
                    .build();
                ui.slider_f32(im_str!("Scale Substrate x"),
                &mut state.scale_substrate_x,
                0.0,
                1.0)
                    .build();
                ui.slider_f32(im_str!("Scale Substrate y"),
                &mut state.scale_substrate_y,
                0.0,
                1.0)
                    .build();
                ui.slider_f32(im_str!("Scale Substrate z"),
                &mut state.scale_substrate_z,
                0.0,
                1.0)
                    .build();
            }

            if ui.collapsing_header(im_str!("CPPN")).build() {
                ui.slider_f32(im_str!("Link Expression Min"),
                &mut state.link_expression_min,
                -1.0,
                1.0)
                    .build();
                ui.slider_f32(im_str!("Link Expression Max"),
                &mut state.link_expression_max,
                -1.0,
                1.0)
                    .build();

                ui.slider_f32(im_str!("Link Weight Range (bipolar)"),
                &mut state.link_weight_range,
                0.1,
                5.0)
                    .build();
                ui.slider_f32(im_str!("Link Weight Creation Sigma"),
                &mut state.link_weight_creation_sigma,
                0.01,
                1.0)
                    .build();
                ui.slider_f32(im_str!("Weight Perturbance Sigma"),
                &mut state.weight_perturbance_sigma,
                0.0,
                1.0)
                    .build();
            }

            if ui.collapsing_header(im_str!("Neighbor Matching")).build() {
                ui.checkbox(im_str!("Edge Weight Scoring"), &mut state.nm_edge_score);
                ui.slider_i32(im_str!("Iterations"), &mut state.nm_iters, 1, 1000).build();
                ui.slider_f32(im_str!("Eps"), &mut state.nm_eps, 0.0, 1.0).build();
            }

            if ui.collapsing_header(im_str!("Mutation")).build() {
                ui.slider_f32(im_str!("Mutation Rate"),
                &mut state.mutate_element_prob,
                0.0,
                1.0)
                    .build();
                ui.slider_i32(im_str!("Weights"), &mut state.mutate_weights, 1, 100).build();
                ui.slider_i32(im_str!("Add Node"), &mut state.mutate_add_node, 0, 100).build();
                ui.slider_i32(im_str!("Drop Node"), &mut state.mutate_drop_node, 0, 100).build();
                ui.slider_i32(im_str!("Modify Node"),
                &mut state.mutate_modify_node,
                0,
                100)
                    .build();
                ui.slider_i32(im_str!("Connect"), &mut state.mutate_connect, 0, 100).build();
                ui.slider_i32(im_str!("Disconnect"), &mut state.mutate_disconnect, 0, 100).build();
                ui.slider_i32(im_str!("Sym Join"), &mut state.mutate_symmetric_join, 0, 100).build();
                ui.slider_i32(im_str!("Sym Fork"), &mut state.mutate_symmetric_fork, 0, 100).build();
                ui.slider_i32(im_str!("Sym Connect"), &mut state.mutate_symmetric_connect, 0, 100).build();
            }
            if ui.collapsing_header(im_str!("Global Mutation")).build() {
                ui.slider_f32(im_str!("Global Mutation Rate"), &mut state.global_mutation_rate,
                0.0,
                1.0)
                    .build();
                ui.slider_f32(im_str!("Element Mutation Rate"), &mut state.global_element_mutation,
                0.0,
                1.0)
                    .build();

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
                let (behavior, connection_cost, sat) = expression.express(genome,
                                                                          &mut network_builder,
                                                                          substrate_config);

                // Evaluate domain specific fitness
                let domain_fitness = fitness_eval.fitness(network_builder.network());

                Fitness {
                    domain_fitness: domain_fitness,
                    behavioral_diversity: 0.0, // will be calculated in `population_metric`
                    connection_cost: connection_cost,
                    behavior: behavior,
                    age_diversity: 0.0,  // will be calculated in `population_metric`
                    saturation: sat.sum(),
                    complexity: genome.complexity(), 
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
        let mut substrate: Substrate<Position3d, Neuron> = Substrate::new();
        let node_count = domain_fitness_eval.target_graph_node_count();
        println!("{:?}", node_count);

        let input_nodeset = NodeSet::single(0);
        let hidden_nodeset = NodeSet::single(1);
        let output_nodeset = NodeSet::single(2);

        let nodeset_links = &[(input_nodeset, hidden_nodeset), (hidden_nodeset, output_nodeset), (input_nodeset, output_nodeset)];

        // Input layer
        {
            let z = 1.0;
            //let y = 0.0;
            //for x in DistributeInterval::new(node_count.inputs, -1.0 * node_count.inputs as f64 / 2.0, 1.0 * node_count.inputs as f64 / 2.0) {
            for x in DistributeInterval::new(node_count.inputs, -1.0, 1.0) {
                let y = 0.0; //0.1 * (1.0 - x.powi(8));
                substrate.add_node(Position3d::new(x, y, z),
                Neuron::Input,
                input_nodeset);
            }
        }

        // Hidden
        {
            let z = 0.0;
            //let y = 0.0;
            //for x in DistributeInterval::new(node_count.inputs, -1.0 * node_count.hidden as f64 / 2.0, 1.0 * node_count.hidden as f64 / 2.0) {
            for x in DistributeInterval::new(node_count.inputs, -1.0, 1.0) {
                //let y = (1.0 - x.powi(8));
                let y = 0.0;
                substrate.add_node(Position3d::new(x, y, z),
                Neuron::Hidden,
                hidden_nodeset);
            }
        }

        // Outputs
        {
            let z = -1.0;
            //let y = 0.0;
            //let mut z = DistributeInterval::new(node_count.outputs, -0.1, 0.1);
            //for x in DistributeInterval::new(node_count.outputs, -1.0 * node_count.outputs as f64 / 2.0, 1.0* node_count.outputs as f64 / 2.0) {
            for x in DistributeInterval::new(node_count.outputs, -1.0, 1.0) {
                //let y = -0.1 * (1.0 - x.powi(8));
                let y = 0.0;
                //substrate.add_node(Position3d::new(x, y, -z.next().unwrap()),
                substrate.add_node(Position3d::new(x, y, z),
                Neuron::Output,
                output_nodeset);
            }
        }

        let mut evo_config = EvoConfig {
            mu: 50,
            lambda: 25,
            k: 2,
            objectives: vec![0,1,2,3,4,5],
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
                //GeometricActivationFunction::Gaussian,
                GeometricActivationFunction::BipolarGaussian,
                GeometricActivationFunction::BipolarSigmoid,
                GeometricActivationFunction::Sine,
                //GeometricActivationFunction::Absolute,
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

        let substrate_config = substrate.to_configuration(nodeset_links);

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

            rated.select(evo_config.mu,
                         &evo_config.objectives,
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
            running: false,
            recalc_fitness: false,
            // recalc_substrate
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

            mutate_symmetric_join: reproduction.mating_method_weights.mutate_symmetric_join as i32,
            mutate_symmetric_fork: reproduction.mating_method_weights.mutate_symmetric_fork as i32,
            mutate_symmetric_connect: reproduction.mating_method_weights.mutate_symmetric_connect as i32,

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
        };

        let mut program_substrate: Option<glium::Program> = None;
        let mut program_vertex: Option<glium::Program> = None;

        loop {
            {
                support.render(CLEAR_COLOR, |display, imgui, renderer, target, delta_f| {

                    if program_substrate.is_none() {
                        program_substrate = Some(program!(display,
                                                          140 => {
                                                              vertex: "
                    #version 140
                    uniform mat4 matrix;
                    uniform mat4 perspective;
                    in vec3 position;
                    in vec4 color;
                    out vec4 fl_color;
                    void main() {
                        gl_Position = perspective * matrix * vec4(position, 1.0);
                        fl_color = color;
                    }
                ",

                fragment: "
                    #version 140
                    in vec4 fl_color;
                    out vec4 color;
                    void main() {
                        color = fl_color;
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
                    in vec3 position;
                    in vec4 color;
                    out vec4 fl_color;
                    void main() {
                        gl_Position = matrix * vec4(position, 1.0);
                        fl_color = color;
                    }
                ",

                fragment: "
                    #version 140
                    in vec4 fl_color;
                    out vec4 color;
                    void main() {
                        color = fl_color;
                    }
                "
                                                       },
                                                       ).unwrap());
                    }

                    const N: usize = 4;

                    let (width, height) = target.get_dimensions();
                    match state.view { 
                        ViewMode::BestDetailed => {
                            let best_ind = &parents.individuals()[best_individual_i];

                            let (substrate_width, substrate_height) = (400, 400);

                            render_graph(display, target, best_ind.genome(), &expression, program_substrate.as_ref().unwrap(), &state, &substrate_config,
                            glium::Rect {left: 0, bottom: 0, width: substrate_width, height: substrate_height}, 2.0, 5.0);


                            render_cppn(display, target, best_ind.genome(), &expression, program_vertex.as_ref().unwrap(), &state, &substrate_config,
                            glium::Rect {left: substrate_width, bottom: 0, width: width-substrate_width, height: height});
                        }
                        ViewMode::Overview => { 
                            let indiv = parents.individuals();
                            let mut indices: Vec<_> = (0..indiv.len()).collect();
                            indices.sort_by(|&i, &j| {
                                match indiv[i].fitness().domain_fitness.partial_cmp(&indiv[j].fitness().domain_fitness).unwrap().reverse() {
                                    Ordering::Equal => {
                                        indiv[i].fitness().behavioral_diversity.partial_cmp(&indiv[j].fitness().behavioral_diversity).unwrap().reverse()
                                    }
                                    a => a
                                }
                            });
                            let mut i = 0;
                            'outer: for y in 0..N {
                                for x in 0..(2*N) {
                                    if i >= indiv.len() {
                                        break 'outer;
                                    }
                                    let rect = glium::Rect {left: (x as u32)*width/(2*N as u32), bottom: (y as u32)*height/(2*N as u32), width: width/(2*N as u32), height: height/(2*N as u32)};
                                    let genome = indiv[indices[i]].genome();
                                    render_cppn(display, target, genome, &expression, program_vertex.as_ref().unwrap(), &state, &substrate_config, rect);
                                    i += 1;
                                }
                            }

                            let mut i = 0;
                            'outer: for y in N..(2*N) {
                                for x in 0..(2*N) {
                                    if i >= indiv.len() {
                                        break 'outer;
                                    }
                                    let rect = glium::Rect {left: (x as u32)*width/(2*N as u32), bottom: (y as u32)*height/(2*N as u32), width: width/(2*N as u32), height: height/(2*N as u32)};
                                    let genome = indiv[indices[i]].genome();
                                    render_graph(display, target, genome, &expression, program_substrate.as_ref().unwrap(), &state, &substrate_config, rect, 1.0, 2.5);
                                    i += 1;
                                }
                            }

                        }

                        ViewMode::CppnOverview | ViewMode::GraphOverview => {
                            let indiv = parents.individuals();
                            let mut indices: Vec<_> = (0..indiv.len()).collect();
                            indices.sort_by(|&i, &j| {
                                match indiv[i].fitness().domain_fitness.partial_cmp(&indiv[j].fitness().domain_fitness).unwrap().reverse() {
                                    Ordering::Equal => {
                                        indiv[i].fitness().behavioral_diversity.partial_cmp(&indiv[j].fitness().behavioral_diversity).unwrap().reverse()
                                    }
                                    a => a
                                }
                            });
                            let mut i = 0;
                            'outer: for y in 0..(2*N) {
                                for x in 0..(2*N) {
                                    if i >= indiv.len() {
                                        break 'outer;
                                    }
                                    let rect = glium::Rect {left: (x as u32)*width/(2*N as u32), bottom: (y as u32)*height/(2*N as u32), width: width/(2*N as u32), height: height/(2*N as u32)};
                                    let genome = indiv[indices[i]].genome();
                                    if let ViewMode::CppnOverview = state.view {
                                        render_cppn(display, target, genome, &expression, program_vertex.as_ref().unwrap(), &state, &substrate_config, rect);
                                    } else {
                                        render_graph(display, target, genome, &expression, program_substrate.as_ref().unwrap(), &state, &substrate_config, rect, 1.0, 2.5);
                                    }
                                    i += 1;
                                }
                            }
                        }
                    }

                    let ui = imgui.frame(width, height, delta_f);
                    gui(&ui, &mut state, &parents);
                    renderer.render(target, ui).unwrap();
                }
                );

                let active = support.update_events();
                if !active {
                    break;
                }

                match state.action {
                    Action::ExportBest => {
                        println!("Export best");
                        println!("Iteration: {}", state.iteration);

                        let best = &parents.individuals()[best_individual_i];

                        let filename = format!("best.{}.{}.gml", state.iteration, (best.fitness().domain_fitness * 1000.0) as usize);
                        println!("filename: {}", filename);

                        let mut file = File::create(&filename).unwrap();
                        let mut network_builder = GMLNetworkBuilder::new();
                        network_builder.set_writer(&mut file);
                        network_builder.begin();
                        let (_behavior, _connection_cost, _) = expression.express(best.genome(),
                        &mut network_builder,
                        &substrate_config);
                        network_builder.end();
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
                        best_fitness = parents.individuals()[best_individual_i].fitness().domain_fitness;
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
                    _ => {
                    }
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

                reproduction.mating_method_weights.mutate_symmetric_join = state.mutate_symmetric_join as u32;
                reproduction.mating_method_weights.mutate_symmetric_fork = state.mutate_symmetric_fork as u32;
                reproduction.mating_method_weights.mutate_symmetric_connect = state.mutate_symmetric_connect as u32;

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
                    parents = next_gen.select(evo_config.mu,
                                              &evo_config.objectives,
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

            if state.best_fitness >= 0.99 {
                state.running = false;
            }

            if state.running {
                // create next generation
                state.iteration += 1;
                let offspring = parents.reproduce(&mut rng,
                                                  evo_config.lambda,
                                                  evo_config.k,
                                                  &|rng, p1, p2| reproduction.mate(rng, p1, p2, state.iteration));
                let mut next_gen = 
                if state.global_mutation_rate > 0.0 {
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
                parents = next_gen.select(evo_config.mu,
                                          &evo_config.objectives,
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
