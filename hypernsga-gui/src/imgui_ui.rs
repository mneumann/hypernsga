use super::{Action, State, ViewMode};
use hypernsga::cppn::G;
use hypernsga::fitness::Fitness;
use imgui::*;
use nsga2::population::RankedPopulation;

pub fn gui<'a>(ui: &Ui<'a>, state: &mut State, population: &RankedPopulation<G, Fitness>) {
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

            if ui.collapsing_header(im_str!("General")).build() {
                if ui.small_button(im_str!("Recalc Fitness")) {
                    state.recalc_fitness = true;
                }
                if ui.small_button(im_str!("Export Best")) {
                    state.action = Action::ExportBest;
                }
                if ui.small_button(im_str!("Reset")) {
                    state.action = Action::ResetNet;
                }

                ui.slider_float(
                    im_str!("Stop when Fitness"),
                    &mut state.stop_when_fitness_above,
                    0.9,
                    1.0,
                )
                .build();
                ui.checkbox(im_str!("Enable Autoreset"), &mut state.auto_reset_enable);
                ui.slider_int(
                    im_str!("Autoreset after"),
                    &mut state.auto_reset,
                    100,
                    10000,
                )
                .build();
                ui.text(im_str!("Autoreset counter: {}", state.auto_reset_counter));

                let views = &[
                    im_str!("detailed"),
                    im_str!("multi cppn"),
                    im_str!("multi graph"),
                    im_str!("overview"),
                ];
                let mut current: i32 = match state.view {
                    ViewMode::BestDetailed => 0,
                    ViewMode::CppnOverview => 1,
                    ViewMode::GraphOverview => 2,
                    ViewMode::Overview => 3,
                };
                if ui.combo(im_str!("view"), &mut current, views, 4) {
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
                let best_domain = population
                    .individuals()
                    .iter()
                    .max_by_key(|ind| (ind.fitness().domain_fitness * 1_000_000.0) as usize)
                    .unwrap();
                let worst_domain = population
                    .individuals()
                    .iter()
                    .min_by_key(|ind| (ind.fitness().domain_fitness * 1_000_000.0) as usize)
                    .unwrap();
                ui.text(im_str!(
                    "Best Domain Fitness: {:.3}",
                    best_domain.fitness().domain_fitness
                ));
                ui.text(im_str!(
                    "Age of Best: {}",
                    best_domain.genome().age(state.iteration)
                ));

                ui.text(im_str!(
                    "Worst Domain Fitness: {:.3}",
                    worst_domain.fitness().domain_fitness
                ));
                ui.text(im_str!(
                    "Age of Worst: {}",
                    worst_domain.genome().age(state.iteration)
                ));
            }

            if ui.collapsing_header(im_str!("History")).build() {
                let fitness_histogram: Vec<_> = state
                    .best_fitness_history
                    .iter()
                    .map(|&(_i, f)| f as f32)
                    .collect();
                PlotLines::new(im_str!("performance"), &fitness_histogram)
                    .overlay_text(im_str!("Domain Fitness"))
                    .graph_size(ImVec2::new(400.0, 50.0))
                    .scale_min(0.0)
                    .scale_max(1.0)
                    .build();
            }
            if ui.collapsing_header(im_str!("Objectives")).build() {
                ui.checkbox(
                    im_str!("Behavioral Diversity"),
                    &mut state.objectives_use_behavioral,
                );
                ui.checkbox(im_str!("Connection Cost"), &mut state.objectives_use_cct);
                ui.checkbox(im_str!("Age Diversity"), &mut state.objectives_use_age);
                ui.checkbox(im_str!("Saturation"), &mut state.objectives_use_saturation);
                ui.checkbox(im_str!("Complexity"), &mut state.objectives_use_complexity);
            }
            if ui.collapsing_header(im_str!("Population Settings")).build() {
                ui.slider_int(im_str!("Population Size"), &mut state.mu, state.k, 1000)
                    .build();
                ui.slider_int(im_str!("Offspring Size"), &mut state.lambda, 1, 1000)
                    .build();
            }

            if ui.collapsing_header(im_str!("Selection")).build() {
                ui.slider_int(im_str!("Tournament Size"), &mut state.k, 1, state.mu)
                    .build();
                ui.slider_float(
                    im_str!("NSGP Objective Epsilon"),
                    &mut state.nsgp_objective_eps,
                    0.0,
                    1.0,
                )
                .build();
            }

            if ui.collapsing_header(im_str!("View")).build() {
                ui.slider_float(
                    im_str!("Rotate Substrate x"),
                    &mut state.rotate_substrate_x,
                    0.0,
                    360.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Rotate Substrate y"),
                    &mut state.rotate_substrate_y,
                    0.0,
                    360.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Rotate Substrate z"),
                    &mut state.rotate_substrate_z,
                    0.0,
                    360.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Scale Substrate x"),
                    &mut state.scale_substrate_x,
                    0.0,
                    1.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Scale Substrate y"),
                    &mut state.scale_substrate_y,
                    0.0,
                    1.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Scale Substrate z"),
                    &mut state.scale_substrate_z,
                    0.0,
                    1.0,
                )
                .build();
            }

            if ui.collapsing_header(im_str!("CPPN")).build() {
                ui.slider_float(
                    im_str!("Link Expression Min"),
                    &mut state.link_expression_min,
                    -1.0,
                    1.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Link Expression Max"),
                    &mut state.link_expression_max,
                    -1.0,
                    1.0,
                )
                .build();

                ui.slider_float(
                    im_str!("Link Weight Range (bipolar)"),
                    &mut state.link_weight_range,
                    0.1,
                    5.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Link Weight Creation Sigma"),
                    &mut state.link_weight_creation_sigma,
                    0.01,
                    1.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Weight Perturbance Sigma"),
                    &mut state.weight_perturbance_sigma,
                    0.0,
                    1.0,
                )
                .build();
            }

            if ui.collapsing_header(im_str!("Neighbor Matching")).build() {
                ui.checkbox(im_str!("Edge Weight Scoring"), &mut state.nm_edge_score);
                ui.slider_int(im_str!("Iterations"), &mut state.nm_iters, 1, 1000)
                    .build();
                ui.slider_float(im_str!("Eps"), &mut state.nm_eps, 0.0, 1.0)
                    .build();
            }

            if ui.collapsing_header(im_str!("Mutation")).build() {
                ui.slider_float(
                    im_str!("Mutation Rate"),
                    &mut state.mutate_element_prob,
                    0.0,
                    1.0,
                )
                .build();
                ui.slider_int(im_str!("Weights"), &mut state.mutate_weights, 1, 100)
                    .build();
                ui.slider_int(im_str!("Add Node"), &mut state.mutate_add_node, 0, 100)
                    .build();
                ui.slider_int(im_str!("Drop Node"), &mut state.mutate_drop_node, 0, 100)
                    .build();
                ui.slider_int(
                    im_str!("Modify Node"),
                    &mut state.mutate_modify_node,
                    0,
                    100,
                )
                .build();
                ui.slider_int(im_str!("Connect"), &mut state.mutate_connect, 0, 100)
                    .build();
                ui.slider_int(im_str!("Disconnect"), &mut state.mutate_disconnect, 0, 100)
                    .build();
                ui.slider_int(
                    im_str!("Sym Join"),
                    &mut state.mutate_symmetric_join,
                    0,
                    100,
                )
                .build();
                ui.slider_int(
                    im_str!("Sym Fork"),
                    &mut state.mutate_symmetric_fork,
                    0,
                    100,
                )
                .build();
                ui.slider_int(
                    im_str!("Sym Connect"),
                    &mut state.mutate_symmetric_connect,
                    0,
                    100,
                )
                .build();
            }
            if ui.collapsing_header(im_str!("Global Mutation")).build() {
                ui.slider_float(
                    im_str!("Global Mutation Rate"),
                    &mut state.global_mutation_rate,
                    0.0,
                    1.0,
                )
                .build();
                ui.slider_float(
                    im_str!("Element Mutation Rate"),
                    &mut state.global_element_mutation,
                    0.0,
                    1.0,
                )
                .build();
            }

            // ui.separator();
            // let mouse_pos = ui.imgui().mouse_pos();
            // ui.text(im_str!("Mouse Position: ({:.1},{:.1})", mouse_pos.0, mouse_pos.1));
        })
}
