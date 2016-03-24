use cppn_ext::cppn::{Cppn, CppnNode};
use cppn_ext::activation_function::ActivationFunction;
pub use cppn_ext::activation_function::GeometricActivationFunction;
use weight::{Weight, WeightRange, WeightPerturbanceMethod, gaussian};
use substrate::{Position, SubstrateConfiguration};
use behavioral_bitvec::BehavioralBitvec;
use genome::Genome;
use nsga2::driver::Driver;
use nsga2::population::RatedPopulation;
use nsga2::selection::{SelectNSGA, SelectNSGP};
use fitness::{Behavior, Fitness, DomainFitness};
use rand::Rng;
use mating::{MatingMethod, MatingMethodWeights};
use prob::Prob;
use network_builder::NetworkBuilder;
use std::marker::PhantomData;

pub type CppnGenome<AF> where AF: ActivationFunction = Genome<CppnNode<AF>>;

const CPPN_OUTPUT_LINK_WEIGHT1: usize = 0;
const CPPN_OUTPUT_LINK_EXPRESSION: usize = 1;
const CPPN_OUTPUT_LINK_WEIGHT2: usize = 2;
const CPPN_OUTPUT_NODE_WEIGHT: usize = 3;

/// Develops a network out of the CPPN
///
/// Returns the Behavior and Connection Cost of the developed network

fn develop_cppn<'a, P, AF, T, V>(cppn: &mut Cppn<CppnNode<AF>, Weight, ()>,
                                 substrate_config: &SubstrateConfiguration<'a, P, T>,
                                 visitor: &mut V,
                                 leo_threshold: f64)
                                 -> (Behavior, f64)
    where P: Position,
          AF: ActivationFunction,
          V: NetworkBuilder<POS = P, NT = T>
{
    // our CPPN has at least two outputs: link weight 1, link expression. optional: link weight 2, node weight
    assert!(cppn.output_count() >= 2);
    assert!(cppn.input_count() == P::DIMENSIONS * 2);

    let nodes = substrate_config.nodes();
    let links = substrate_config.links();
    let null_position = substrate_config.null_position();

    let mut behavior = Behavior {
        bv_link_weight1:    BehavioralBitvec::new(links.len()),
        bv_link_weight2:    BehavioralBitvec::new(links.len()),
        bv_link_expression: BehavioralBitvec::new(links.len()),
        bv_node_weight:     BehavioralBitvec::new(nodes.len()),
    };
    //let mut bitvec = BehavioralBitvec::new(cppn.output_count() * (nodes.len() + links.len()));

    let mut connection_cost = 0.0;

    // First visit all nodes

    for node in nodes.iter() {
        let inputs = [node.position.coords(), null_position.coords()];
        cppn.process(&inputs[..]);

        //let link_weight1 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT1).unwrap();
        //let link_expression = cppn.read_output(CPPN_OUTPUT_LINK_EXPRESSION).unwrap();

        // link_weight2 and node_weight are optional. in case they don't exist, set them to -1.0
        // (-1.0 results in the behavioral bitvector set to all 0)
        //let link_weight2 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT2).unwrap_or(-1.0);

        let node_weight = cppn.read_output(CPPN_OUTPUT_NODE_WEIGHT).unwrap_or(-1.0);

        behavior.bv_node_weight.push(node_weight);

        visitor.add_node(node, node_weight)
    }

    for &(source_node, target_node) in links.iter() {
        let inputs = [source_node.position.coords(), target_node.position.coords()];
        cppn.process(&inputs[..]);

        let link_weight1 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT1).unwrap();
        let link_expression = cppn.read_output(CPPN_OUTPUT_LINK_EXPRESSION).unwrap();

        // link_weight2 and node_weight are optional. in case they don't exist, set them to -1.0
        // (-1.0 results in the behavioral bitvector set to all 0)
        let link_weight2 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT2).unwrap_or(-1.0);
        //let node_weight = cppn.read_output(CPPN_OUTPUT_NODE_WEIGHT).unwrap_or(-1.0);

        behavior.bv_link_weight1.push(link_weight1);
        behavior.bv_link_expression.push(link_expression);
        behavior.bv_link_weight2.push(link_weight2);

        if link_expression > leo_threshold {
            let distance_sq = source_node.position.distance_square(&target_node.position);
            debug_assert!(distance_sq >= 0.0);
            connection_cost += distance_sq;
            visitor.add_link(source_node, target_node, link_weight1, link_weight2);
        }
    }

    return (behavior, connection_cost);
}

/// Determines the domain fitness of `G`

pub struct CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G> + 'a,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync,
          G: Sync
{
    pub mating_method_weights: MatingMethodWeights,
    pub activation_functions: Vec<GeometricActivationFunction>,
    pub mutate_element_prob: Prob,
    pub weight_perturbance: WeightPerturbanceMethod,
    pub link_weight_range: WeightRange,
    pub link_weight_creation_sigma: f64,

    pub mutate_add_node_random_link_weight: bool,
    pub mutate_drop_node_tournament_k: usize,
    pub mutate_modify_node_tournament_k: usize,

    pub mate_retries: usize,

    pub link_expression_threshold: f64,

    pub substrate_configuration: SubstrateConfiguration<'a, P, T>,
    pub domain_fitness: &'a DOMFIT,
    pub _netbuilder: PhantomData<NETBUILDER>,

    /// Create genomes which are already connected
    pub start_connected: bool,

    /// The initial connections are random within this range
    pub start_link_weight_range: WeightRange,

    /// Create a gaussian seed node for the symmetry of d-th axis with the given weight
    /// vec![Some(1.0), None, Some(2.0)] would for example create a node for symmetry of x-axis and
    /// z-axis.
    pub start_symmetry: Vec<Option<f64>>,

    /// Create `start_initial_nodes` random nodes for each genome. This can be useful,
    /// as we have a pretty low probability for adding a node.
    pub start_initial_nodes: usize,
}

impl<'a, DOMFIT, G, P, T, NETBUILDER> CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G>,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync,
          G: Sync
{
    fn random_hidden_node<R>(&self, rng: &mut R) -> CppnNode<GeometricActivationFunction>
        where R: Rng
    {
        let af = *rng.choose(&self.activation_functions).unwrap();
        CppnNode::hidden(af)
    }
}

impl<'a, DOMFIT, G, P, T, NETBUILDER> Driver for CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G>,
          G: Sync,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync,
          G: Sync
{
    type GENOME = CppnGenome<GeometricActivationFunction>;
    type FIT = Fitness;
    type SELECTION = SelectNSGP;

    /// Creates a random individual for use by the start generation.
    ///
    /// We start from a minimal topology.
    /// If `self.start_connected` is `true`, we add some initial connections.

    fn random_genome<R>(&self, rng: &mut R) -> Self::GENOME
        where R: Rng
    {
        let mut genome = Self::GENOME::new();

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        // for every dimension we use two inputs e.g. (x1, x2), each for the other end of the
        // connection

        for _d in 0..P::DIMENSIONS {
            let inp1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
            let inp2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));

            inputs.push(inp1);
            inputs.push(inp2);
        }

        // 1 bias node (constant input of 1.0)
        let bias = genome.add_node(CppnNode::bias(GeometricActivationFunction::Constant1));
        inputs.push(bias);

        // 4 outputs (t,ex,w,r)
        let out_t = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        let out_ex = genome.add_node(CppnNode::output(GeometricActivationFunction::Linear));
        let out_w = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        let out_r = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        outputs.push(out_t);
        outputs.push(out_ex);
        outputs.push(out_w);
        outputs.push(out_r);

        // make those nodes above immutable for mutation and crossover, as we need them to develop
        // the CPPN.
        genome.protect_nodes();

        for d in 0..P::DIMENSIONS {
            if let &Some(w) = self.start_symmetry.get(d).unwrap_or(&None) {
                let sym =
                    genome.add_node(CppnNode::hidden(GeometricActivationFunction::BipolarGaussian));
                let inp1 = inputs[d * 2];
                let inp2 = inputs[d * 2 + 1];
                genome.add_link(inp1, sym, self.link_weight_range.clip_weight(Weight(-w)));
                genome.add_link(inp2, sym, self.link_weight_range.clip_weight(Weight(w)));
                // XXX: should we really connect the sym node as well?
                inputs.push(sym);
            }
        }

        // Make sure that at least every input and every output is connected.
        if self.start_connected {
            let mut connections: Vec<(usize, usize)> = Vec::new();

            // make a connection from every input to a random output
            for (inp, _) in inputs.iter().enumerate() {
                let outp = rng.gen_range(0, outputs.len());
                connections.push((inp, outp));
            }
            // make a connection from every output to a random input
            for (outp, _) in outputs.iter().enumerate() {
                let inp = rng.gen_range(0, inputs.len());
                connections.push((inp, outp));
            }
            // remove duplicates
            connections.sort();
            connections.dedup();

            println!("connections: {:?}", connections);

            // and add the connections to the genome
            for (inp, outp) in connections {
                genome.add_link(inputs[inp],
                                outputs[outp],
                                self.start_link_weight_range.random_weight(rng));
            }
        }

        for _ in 0..self.start_initial_nodes {
            let _ = genome.add_node(self.random_hidden_node(rng));
        }

        genome
    }

    fn fitness(&self, ind: &Self::GENOME) -> Self::FIT {
        let mut cppn = Cppn::new(ind.network());
        let mut net_builder = NETBUILDER::new();

        let (behavior, connection_cost) = develop_cppn(&mut cppn,
                                                       &self.substrate_configuration,
                                                       &mut net_builder,
                                                       self.link_expression_threshold);

        // Evaluate domain specific fitness
        let domain_fitness = self.domain_fitness.fitness(net_builder.network());

        Fitness {
            domain_fitness: domain_fitness,
            behavioral_diversity: 0.0, // will be calculated in `population_metric`
            connection_cost: connection_cost,
            behavior: behavior,
        }
    }

    fn mate<R>(&self, rng: &mut R, parent1: &Self::GENOME, parent2: &Self::GENOME) -> Self::GENOME
        where R: Rng
    {
        let mut offspring = parent1.clone();

        for _ in 0..self.mate_retries + 1 {
            let modified = match MatingMethod::random_with(&self.mating_method_weights, rng) {
                MatingMethod::MutateAddNode => {
                    let link_weight = if self.mutate_add_node_random_link_weight {
                        Some(self.link_weight_range.random_weight(rng))
                    } else {
                        // duplicate existing node weight
                        None
                    };
                    let hidden_node = self.random_hidden_node(rng);
                    offspring.mutate_add_node(hidden_node, link_weight, rng)
                }
                MatingMethod::MutateDropNode => {
                    offspring.mutate_drop_node(self.mutate_drop_node_tournament_k, rng)
                }
                MatingMethod::MutateModifyNode => {
                    let hidden_node = self.random_hidden_node(rng);
                    offspring.mutate_modify_node(hidden_node,
                                                 self.mutate_modify_node_tournament_k,
                                                 rng)
                }
                MatingMethod::MutateConnect => {
                    let link_weight =
                        self.link_weight_range
                            .clip_weight(Weight(gaussian(self.link_weight_creation_sigma, rng)));
                    offspring.mutate_connect(link_weight, rng)
                }
                MatingMethod::MutateDisconnect => offspring.mutate_disconnect(rng),
                MatingMethod::MutateWeights => {
                    let modifications = offspring.mutate_weights(self.mutate_element_prob,
                                                                 &self.weight_perturbance,
                                                                 &self.link_weight_range,
                                                                 rng);
                    modifications != 0
                }
                MatingMethod::CrossoverWeights => {
                    let modifications = offspring.crossover_weights(parent2, rng);
                    modifications != 0
                }
            };

            if modified {
                break;
            }
        }

        warn!("mate(): Genome was NOT modified!");

        return offspring;
    }

    fn population_metric(&self, population: &mut RatedPopulation<Self::GENOME, Self::FIT>) {
        // Determine the behavioral_diversity as average hamming distance to all other individuals.
        // hamming distance is symmetric.

        let n = population.len();

        // reset all behavioral_diversity values to 0
        for i in 0..n {
            population.fitness_mut(i).behavioral_diversity = 0.0;
        }

        for i in 0..n {
            // determine  hehavioral diversity for `i`.
            let mut diversity_i = 0.0;

            // XXX: parallelize this loop
            for j in i + 1..n {
                let distance = population.fitness(i).behavior.weighted_distance(&population.fitness(j).behavior);
                diversity_i += distance;
                population.fitness_mut(j).behavioral_diversity += distance;
            }

            population.fitness_mut(i).behavioral_diversity = diversity_i;
        }
    }
}
