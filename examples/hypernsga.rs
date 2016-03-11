extern crate neat;
extern crate rand;
extern crate graph_neighbor_matching;
extern crate graph_io_gml;
extern crate closed01;
extern crate petgraph;
extern crate cppn;
extern crate asexp;
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate nsga2;

mod common;
mod config;

use neat::population::{UnratedPopulation, Individual, Population, PopulationWithRank};
use neat::niching::NicheRunner;
use neat::traits::FitnessEval;
use neat::genomes::acyclic_network::{Genome, GlobalCache, GlobalInnovationCache, Mater,
ElementStrategy};
use neat::fitness::Fitness;
use graph_neighbor_matching::graph::{GraphBuilder, OwnedGraph};
use graph_neighbor_matching::NodeColorWeight;
use rand::Rng;
use std::marker::PhantomData;
use common::{load_graph, Neuron, convert_neuron_from_str, GraphSimilarity, NodeCount, write_gml, NodeLabel, genome_to_petgraph, write_gml_petgraph,
write_petgraph_as_dot};
use cppn::cppn::{Cppn, CppnNode, CppnNodeKind};
use cppn::activation_function::GeometricActivationFunction;
use cppn::substrate::{Substrate, Layer};
use cppn::position::Position2d;
use neat::weight::{Weight, WeightRange};
use neat::distribute::DistributeInterval;
use closed01::Closed01;

type Node = CppnNode<GeometricActivationFunction>;

impl NodeLabel for Node {
    fn node_label(&self, node_idx: usize) -> Option<String> {
        match self.kind { 
            CppnNodeKind::Input => {
                match node_idx {
                    0 => Some("x_s".to_owned()),
                    1 => Some("y_s".to_owned()),
                    2 => Some("x_t".to_owned()),
                    3 => Some("y_t".to_owned()),
                    4 => Some("d".to_owned()),
                    _ => None,
                }
            }
            CppnNodeKind::Bias => { 
                Some("Bias".to_owned())
            }
            CppnNodeKind::Hidden => {
                Some(format!("{:?}", self.activation_function))
            }
            CppnNodeKind::Output => {
                Some("Output".to_owned())
            }
        }

    }
    fn node_shape(&self) -> &'static str {
        match self.kind { 
            CppnNodeKind::Input => "circle",
            CppnNodeKind::Bias => "triangle",
            CppnNodeKind::Hidden => "box",
            CppnNodeKind::Output => "doublecircle",
        }
    }
}

fn generate_substrate(node_count: &NodeCount) -> Substrate<Position2d, Neuron> {
    let mut substrate = Substrate::new();

    let min = -3.0;
    let max = 3.0;

    // XXX: Todo 3d position
    let mut y_iter = DistributeInterval::new(3, min, max); // 3 layers (Input, Hidden, Output)

    let mut input_layer = Layer::new();
    let mut hidden_layer = Layer::new();
    let mut output_layer = Layer::new();

    // Input layer
    {
        let y = y_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.inputs, min, max) {
            input_layer.add_node(Position2d::new(x, y), Neuron::Input);
        }
    }

    // Hidden
    {
        let y = y_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.hidden, min, max) {
            hidden_layer.add_node(Position2d::new(x, y), Neuron::Hidden);
        }
    }

    // Outputs
    {
        let y = y_iter.next().unwrap();
        for x in DistributeInterval::new(node_count.outputs, min, max) {
            output_layer.add_node(Position2d::new(x, y), Neuron::Output);
        }
    }

    let i = substrate.add_layer(input_layer);
    let h = substrate.add_layer(hidden_layer);
    let o = substrate.add_layer(output_layer);

    substrate.add_layer_link(i, h, None);
    substrate.add_layer_link(i, o, None);
    substrate.add_layer_link(h, h, None);
    substrate.add_layer_link(h, o, None);

    return substrate;
}


#[derive(Debug)]
struct FitnessEvaluator {
    sim: GraphSimilarity,
    node_count: NodeCount,
    //weight_threshold: f64,
}

impl FitnessEvaluator {
    fn genome_to_graph(&self, genome: &Genome<Node>) -> OwnedGraph<Neuron> {
        let substrate = generate_substrate(&self.node_count);
        let mut cppn = Cppn::new(genome.network());

        // now develop the cppn. the result is a graph
        let mut builder = GraphBuilder::new();
        for (layer_idx, layer) in substrate.layers().iter().enumerate() { 
            for (node_idx, node) in layer.nodes().iter().enumerate() {
                let _ = builder.add_node((layer_idx, node_idx), node.node_type.clone());
            }
        }

        substrate.each_link(&mut cppn, &mut |link| {
            let mut link_weight = link.outputs[0];
            let mut leo = link.outputs[1]; // link expression output

            if leo > 0.0 {
            //if w0 > w1 {
                let w = link_weight.abs();
               //let w = (w0 + 1.0) / 2.0;
            //if w > 0.5 {//cppn_weight_threshold_min && w < cppn_weight_threshold_max {
                //let normalized = (w - cppn_weight_threshold_min) / (cppn_weight_threshold_max - cppn_weight_threshold_min);
                //if w > 1.0 {
                //    w = 1.0;
                //}

                builder.add_edge(link.source_idx,
                                 link.target_idx,
                                 Closed01::new(w as f32));
                //}
            }
        });

        return builder.graph();
    }
}

impl FitnessEval<Genome<Node>> for FitnessEvaluator {
    // A larger fitness means "better"
    fn fitness(&self, genome: &Genome<Node>) -> Fitness {
        Fitness::new(self.sim.fitness(&self.genome_to_graph(genome)) as f64)
    }
}

struct ES {
    activation_functions: Vec<GeometricActivationFunction>,
    link_weight_range: WeightRange,
    full_link_weight: Weight,
}

impl ElementStrategy<Node> for ES {
    fn link_weight_range(&self) -> WeightRange {
        self.link_weight_range
    }

    fn full_link_weight(&self) -> Weight {
        self.full_link_weight
    }

    fn random_node_type<R: Rng>(&self, rng: &mut R) -> Node {
        CppnNode::hidden(*rng.choose(&self.activation_functions).unwrap())
    }
}

fn run(run_no: usize, cfg: &config::Configuration) {
    let mut rng = rand::thread_rng();

    println!("------------------------");
    println!("RUN #{}", run_no);
    println!("------------------------");

    let target_graph = load_graph(&cfg.target_graph_file(), convert_neuron_from_str);
    let node_count = NodeCount::from_graph(&target_graph);

    let fitness_evaluator = FitnessEvaluator {
        sim: GraphSimilarity {
            target_graph: target_graph,
            edge_score: cfg.edge_score(),
            iters: cfg.neighbormatching_iters(),
            eps: cfg.neighbormatching_eps(),
        },
        node_count: node_count,
    };

    let mut cache = GlobalInnovationCache::new();

    // start with minimal random topology.

    let mut genome = Genome::new();

    // 4 inputs (x1,y1,x2,y2)
    let n_input1 = cache.create_node_innovation();
    let n_input2 = cache.create_node_innovation();
    let n_input3 = cache.create_node_innovation();
    let n_input4 = cache.create_node_innovation();
    genome.add_node(n_input1, CppnNode::input(GeometricActivationFunction::Linear));
    genome.add_node(n_input2, CppnNode::input(GeometricActivationFunction::Linear));
    genome.add_node(n_input3, CppnNode::input(GeometricActivationFunction::Linear));
    genome.add_node(n_input4, CppnNode::input(GeometricActivationFunction::Linear));

    // 2 output (o1, o2)
    let n_output1 = cache.create_node_innovation();
    let n_output2 = cache.create_node_innovation();
    genome.add_node(n_output1, CppnNode::output(GeometricActivationFunction::Gaussian));
    genome.add_node(n_output2, CppnNode::output(GeometricActivationFunction::BipolarSigmoid));

    // 1 bias node
    let n_bias = cache.create_node_innovation();
    genome.add_node(n_bias, CppnNode::bias(GeometricActivationFunction::Constant1));

    // 1 node for x-distance between source and target neuron
    let n_distance_x = cache.create_node_innovation();
    genome.add_node(n_distance_x, CppnNode::hidden(GeometricActivationFunction::BipolarGaussian));
    let n_distance_y = cache.create_node_innovation();
    genome.add_node(n_distance_y, CppnNode::hidden(GeometricActivationFunction::BipolarGaussian));

    let template_genome = genome;

    let mut niche_runner = NicheRunner::new(&fitness_evaluator);

    for _ in 0..cfg.num_niches() {
        let mut pop = UnratedPopulation::new();

        // XXX: probabilistic round
        let niche_size = cfg.population_size() / cfg.num_niches(); 

        for _ in 0..niche_size {
            let mut genome = template_genome.clone();


            genome.add_link(n_input1, n_distance_x, cache.get_or_create_link_innovation(n_input1, n_distance_x), cfg.link_weight_range().clip_weight(Weight(-3.0)));
            genome.add_link(n_input3, n_distance_x, cache.get_or_create_link_innovation(n_input3, n_distance_x), cfg.link_weight_range().clip_weight(Weight(3.0)));

            genome.add_link(n_input2, n_distance_y, cache.get_or_create_link_innovation(n_input2, n_distance_y), cfg.link_weight_range().clip_weight(Weight(-3.0)));
            genome.add_link(n_input4, n_distance_y, cache.get_or_create_link_innovation(n_input4, n_distance_y), cfg.link_weight_range().clip_weight(Weight(3.0)));

            // connect all input/bias nodes to the output node randomly
            /*
            for &from in &[n_input1, n_input2, n_input3, n_input4] { //, n_distance, n_bias] {
                genome.add_link(from, n_output1, cache.get_or_create_link_innovation(from, n_output1), cfg.link_weight_range().random_weight(&mut rng));
                genome.add_link(from, n_output2, cache.get_or_create_link_innovation(from, n_output2), cfg.link_weight_range().random_weight(&mut rng));
            }
            */
            pop.add_unrated_individual(Individual::new_unrated(Box::new(genome.clone())));
        }

        niche_runner.add_unrated_population_as_niche(pop);
    }

    let es = ES {
        activation_functions: vec![
            //GeometricActivationFunction::Linear,
            //GeometricActivationFunction::LinearBipolarClipped,
            GeometricActivationFunction::BipolarGaussian,
            //GeometricActivationFunction::Gaussian,
            GeometricActivationFunction::BipolarSigmoid,
            //GeometricActivationFunction::Absolute,
            GeometricActivationFunction::Sine,
            //GeometricActivationFunction::Cosine
        ],
        link_weight_range: cfg.link_weight_range(),
        full_link_weight: cfg.full_link_weight(),
    };

    let mut mater = Mater {
        p_crossover: cfg.p_crossover(),
        p_crossover_detail_nodes: cfg.probabilistic_crossover_nodes(),
        p_crossover_detail_links: cfg.probabilistic_crossover_links(),
        p_mutate_element: cfg.p_mutate_element(),
        weight_perturbance: cfg.weight_perturbance(),
        mutate_weights: cfg.mutate_method_weighting(),
        global_cache: &mut cache,
        element_strategy: &es,
        _n: PhantomData,
    };

    let mut fitness_log: Vec<f64> = Vec::new();

    while niche_runner.has_next_iteration(cfg.stop_after_iters()) {
        println!("R{:05} iteration: {}", run_no, niche_runner.current_iteration());

        let best_fitness = niche_runner.best_individual().unwrap().fitness().get();;
        println!("R{:05} best fitness: {:2}", run_no, best_fitness); 
        println!("R{:05} num individuals: {}", run_no, niche_runner.num_individuals());

        if best_fitness > cfg.stop_if_fitness_better_than() {
            println!("R{:05} Premature abort.", run_no);
            break;
        }

        let mut top_n_niches = cfg.num_niches();

        niche_runner.reproduce(cfg.population_size(),
        top_n_niches,
        cfg.elite_percentage(),
        cfg.selection_percentage(),
        cfg.compatibility_threshold(),
        cfg.genome_compatibility(),
        &mut mater,
        &mut rng);
    }

    let final_pop = niche_runner.into_ranked_population();

    let best = final_pop.best_individual().unwrap();

    println!("R{:05} best fitness: {:?}", run_no, best.fitness());

    if best.fitness().get() < 0.8 {
        println!("R{:05} no output", run_no);
        return;
    }

    {
        write_gml(&format!("{:05}_best.gml", run_no), &fitness_evaluator.genome_to_graph(best.genome()), &Neuron::node_color_weight);

        // display CPPN
        write_gml_petgraph(&format!("{:05}_best_cppn.gml", run_no), &genome_to_petgraph(best.genome()), &|_| 0.0);
        write_petgraph_as_dot(&format!("{:05}_best_cppn.dot", run_no), &genome_to_petgraph(best.genome()), &|_| 0.0);
    }

    /*
    for (i, ind) in final_pop.individuals().iter().enumerate() {
        println!("individual #{}: {:.3}", i, ind.fitness().get());
        write_gml(&format!("ind_{:03}_{}.gml", i, (ind.fitness().get() * 100.0) as usize), &fitness_evaluator.genome_to_graph(ind.genome()), &Neuron::node_color_weight);
    }
    */

    write_gml(&format!("{:05}_target.gml", run_no), &fitness_evaluator.sim.target_graph, &Neuron::node_color_weight);
}

fn main() {
    env_logger::init().unwrap();

    let cfg = config::Configuration::from_file();
    println!("{:?}", cfg);

    let mut i = 0;
    loop {
        run(i, &cfg);
        i += 1;
    }
}
