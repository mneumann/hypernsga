#[derive(Debug)]
pub enum ViewMode {
    BestDetailed,
    CppnOverview,
    GraphOverview,
    Overview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    ExportBest,
    ResetNet,
}

#[derive(Debug)]
pub struct State {
    pub iteration: usize,
    pub best_fitness: f64,
    pub running: bool,
    pub mu: i32,
    pub lambda: i32,
    pub k: i32,
    pub mutate_add_node: i32,
    pub mutate_drop_node: i32,
    pub mutate_modify_node: i32,
    pub mutate_connect: i32,
    pub mutate_disconnect: i32,
    pub mutate_symmetric_join: i32,
    pub mutate_symmetric_fork: i32,
    pub mutate_symmetric_connect: i32,
    pub mutate_weights: i32,

    pub mutate_element_prob: f32,
    pub nsgp_objective_eps: f32,
    pub weight_perturbance_sigma: f32,
    pub link_weight_range: f32,
    pub link_weight_creation_sigma: f32,

    pub best_fitness_history: Vec<(usize, f64)>,
    // crossover_weights: 0,
    pub nm_edge_score: bool,
    pub nm_iters: i32,
    pub nm_eps: f32,

    pub rotate_substrate_x: f32,
    pub rotate_substrate_y: f32,
    pub rotate_substrate_z: f32,
    pub scale_substrate_x: f32,
    pub scale_substrate_y: f32,
    pub scale_substrate_z: f32,

    pub link_expression_min: f32,
    pub link_expression_max: f32,

    pub recalc_fitness: bool,

    pub objectives_use_behavioral: bool,
    pub objectives_use_cct: bool,
    pub objectives_use_age: bool,
    pub objectives_use_saturation: bool,
    pub objectives_use_complexity: bool,

    pub action: Action,
    pub view: ViewMode,

    pub global_mutation_rate: f32,
    pub global_element_mutation: f32,

    pub auto_reset: i32,
    pub auto_reset_enable: bool,
    pub auto_reset_counter: usize,

    pub stop_when_fitness_above: f32,
    pub enable_stop: bool,
}
