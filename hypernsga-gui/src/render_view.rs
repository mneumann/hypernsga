use super::{render_cppn, render_graph, Transformation, ViewMode};
use glium;
use glium::backend::glutin_backend::GlutinFacade;
use hypernsga::cppn::{Expression, G};
use hypernsga::domain_graph::Neuron;
use hypernsga::fitness::Fitness;
use hypernsga::substrate::{Position3d, SubstrateConfiguration};
use nsga2::population::RankedPopulation;
use std::cmp::Ordering;

pub fn render_view(
    view_mode: ViewMode,
    parents: &RankedPopulation<G, Fitness>,
    best_individual_i: usize,
    display: &GlutinFacade,
    target: &mut glium::Frame,
    expression: &Expression,
    program_substrate: &glium::Program,
    program_vertex: &glium::Program,
    substrate_config: &SubstrateConfiguration<Position3d, Neuron>,
    transformation: &Transformation,
    n: usize,
    width: u32,
    height: u32,
) {
    match view_mode {
        ViewMode::BestDetailed => {
            let best_ind = &parents.individuals()[best_individual_i];

            let (substrate_width, substrate_height) = (400, 400);

            render_graph(
                display,
                target,
                best_ind.genome(),
                expression,
                program_substrate,
                transformation,
                substrate_config,
                glium::Rect {
                    left: 0,
                    bottom: 0,
                    width: substrate_width,
                    height: substrate_height,
                },
                2.0,
                5.0,
            );

            render_cppn(
                display,
                target,
                best_ind.genome(),
                program_vertex,
                glium::Rect {
                    left: substrate_width,
                    bottom: 0,
                    width: width - substrate_width,
                    height: height,
                },
            );
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
                    .reverse()
                {
                    Ordering::Equal => indiv[i]
                        .fitness()
                        .behavioral_diversity
                        .partial_cmp(&indiv[j].fitness().behavioral_diversity)
                        .unwrap()
                        .reverse(),
                    a => a,
                }
            });

            {
                let mut i = 0;
                'outer1: for y in 0..n {
                    for x in 0..(2 * n) {
                        if i >= indiv.len() {
                            break 'outer1;
                        }
                        let rect = glium::Rect {
                            left: (x as u32) * width / (2 * n as u32),
                            bottom: (y as u32) * height / (2 * n as u32),
                            width: width / (2 * n as u32),
                            height: height / (2 * n as u32),
                        };
                        let genome = indiv[indices[i]].genome();
                        render_cppn(display, target, genome, program_vertex, rect);
                        i += 1;
                    }
                }
            }

            {
                let mut i = 0;
                'outer2: for y in n..(2 * n) {
                    for x in 0..(2 * n) {
                        if i >= indiv.len() {
                            break 'outer2;
                        }
                        let rect = glium::Rect {
                            left: (x as u32) * width / (2 * n as u32),
                            bottom: (y as u32) * height / (2 * n as u32),
                            width: width / (2 * n as u32),
                            height: height / (2 * n as u32),
                        };
                        let genome = indiv[indices[i]].genome();
                        render_graph(
                            display,
                            target,
                            genome,
                            expression,
                            program_substrate,
                            transformation,
                            substrate_config,
                            rect,
                            1.0,
                            2.5,
                        );
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
                    .reverse()
                {
                    Ordering::Equal => indiv[i]
                        .fitness()
                        .behavioral_diversity
                        .partial_cmp(&indiv[j].fitness().behavioral_diversity)
                        .unwrap()
                        .reverse(),
                    a => a,
                }
            });
            let mut i = 0;
            'outer: for y in 0..(2 * n) {
                for x in 0..(2 * n) {
                    if i >= indiv.len() {
                        break 'outer;
                    }
                    let rect = glium::Rect {
                        left: (x as u32) * width / (2 * n as u32),
                        bottom: (y as u32) * height / (2 * n as u32),
                        width: width / (2 * n as u32),
                        height: height / (2 * n as u32),
                    };
                    let genome = indiv[indices[i]].genome();
                    if let ViewMode::CppnOverview = view_mode {
                        render_cppn(display, target, genome, program_vertex, rect);
                    } else {
                        render_graph(
                            display,
                            target,
                            genome,
                            expression,
                            program_substrate,
                            transformation,
                            substrate_config,
                            rect,
                            1.0,
                            2.5,
                        );
                    }
                    i += 1;
                }
            }
        }
    }
}
