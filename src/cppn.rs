pub use cppn_ext::cppn::Cppn;
use cppn_ext::cppn::CppnNode;
pub use cppn_ext::cppn::CppnNodeKind;
pub use cppn_ext::activation_function::ActivationFunction;
pub use cppn_ext::activation_function::GeometricActivationFunction;
use weight::{Weight, WeightRange, WeightPerturbanceMethod, gaussian};
use substrate::{Position, SubstrateConfiguration};
use behavioral_bitvec::BehavioralBitvec;
use genome::Genome;
use nsga2::driver::Driver;
use nsga2::population::RatedPopulation;
use nsga2::selection::SelectNSGP;
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

pub struct Saturation {
    pub zero: f64,
    pub low: f64,
    pub high: f64,
}

impl Saturation {
    pub fn sum(&self) -> f64 {
        self.zero + self.low + self.high
    }
}

/// Develops a network out of the CPPN
///
/// Returns the Behavior and Connection Cost of the developed network

fn develop_cppn<P, AF, T, V>(cppn: &mut Cppn<CppnNode<AF>, Weight, ()>,
                             substrate_config: &SubstrateConfiguration<P, T>,
                             visitor: &mut V,
                             link_expression_range: (f64, f64))
                             -> (Behavior, f64, Saturation)
    where P: Position,
          AF: ActivationFunction,
          V: NetworkBuilder<POS = P, NT = T>
{
    // our CPPN has at least two outputs: link weight 1, link expression. optional: link weight 2, node weight
    assert!(cppn.output_count() >= 2);
    assert!(cppn.input_count() == P::dims() * 2);

    let nodes = substrate_config.nodes();
    let links = substrate_config.links();
    let null_position = substrate_config.null_position();

    let mut behavior = Behavior {
        bv_link_weight1: BehavioralBitvec::new(links.len()),
        bv_link_weight2: BehavioralBitvec::new(links.len()),
        bv_link_expression: BehavioralBitvec::new(links.len()),
        bv_node_weight: BehavioralBitvec::new(nodes.len()),
    };

    let mut connection_cost = 0.0;

    // First visit all nodes

    for node in nodes.iter() {
        let inputs = [node.position.coords(), null_position.coords()];
        cppn.process(&inputs[..]);

        // let link_weight1 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT1).unwrap();
        // let link_expression = cppn.read_output(CPPN_OUTPUT_LINK_EXPRESSION).unwrap();

        // link_weight2 and node_weight are optional. in case they don't exist, set them to -1.0
        // (-1.0 results in the behavioral bitvector set to all 0)
        // let link_weight2 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT2).unwrap_or(-1.0);

        let node_weight = cppn.read_output(CPPN_OUTPUT_NODE_WEIGHT).unwrap_or(-1.0);

        behavior.bv_node_weight.push(node_weight);

        visitor.add_node(node, node_weight)
    }

    let mut sat = Saturation {
        zero: 0.0,
        low: 0.0,
        high: 0.0,
    };

    for &(source_node_idx, target_node_idx) in links.iter() {
        let source_node = &nodes[source_node_idx];
        let target_node = &nodes[target_node_idx];
        let inputs = [source_node.position.coords(), target_node.position.coords()];
        cppn.process(&inputs[..]);

        // count all nodes with 0.0 and 1.0 and -1.0 signals
        // these are over/under saturated signals.
        let mut zero_cnt = 0;
        let mut low_cnt = 0;
        let mut high_cnt = 0;
        for &signal in cppn.incoming_signals() {
            if signal == 0.0 {
                zero_cnt += 1;
            } else if signal == -1.0 {
                low_cnt += 1;
            } else if signal == 1.0 {
                high_cnt += 1;
            }
        }
        sat.zero += (zero_cnt as f64) / cppn.incoming_signals().len() as f64;
        sat.low += (low_cnt as f64) / cppn.incoming_signals().len() as f64;
        sat.high += (high_cnt as f64) / cppn.incoming_signals().len() as f64;

        let link_weight1 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT1).unwrap();
        let link_expression = cppn.read_output(CPPN_OUTPUT_LINK_EXPRESSION).unwrap();

        // link_weight2 and node_weight are optional. in case they don't exist, set them to -1.0
        // (-1.0 results in the behavioral bitvector set to all 0)
        let link_weight2 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT2).unwrap_or(-1.0);
        // let node_weight = cppn.read_output(CPPN_OUTPUT_NODE_WEIGHT).unwrap_or(-1.0);

        behavior.bv_link_weight1.push(link_weight1);
        behavior.bv_link_expression.push(link_expression);
        behavior.bv_link_weight2.push(link_weight2);

        if link_expression >= link_expression_range.0 && link_expression <= link_expression_range.1 {
            let distance_sq = source_node.position.distance_square(&target_node.position);
            debug_assert!(distance_sq >= 0.0);
            connection_cost += distance_sq;
            visitor.add_link(source_node, target_node, link_weight1, link_weight2);
        }
    }

    return (behavior, connection_cost, sat);
}

/// Determines the domain fitness of `G`

pub struct CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G> + 'a,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync,
          G: Sync
{
    /// express a link if leo is within [min, max]
    pub link_expression_range: (f64, f64),

    pub substrate_configuration: SubstrateConfiguration<P, T>,
    pub domain_fitness: &'a DOMFIT,
    pub _netbuilder: PhantomData<NETBUILDER>,

    pub reproduction: Reproduction,
    pub random_genome_creator: RandomGenomeCreator,
}

/// XXX: Driver does not correctly set birth_iteration of offspring genomes!!!!
/// XXX: wrong calculation of age_diversity
impl<'a, DOMFIT, G, P, T, NETBUILDER> Driver for CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G>,
          G: Sync,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync
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
        self.random_genome_creator.create::<_, P>(0, rng)
    }

    fn fitness(&self, ind: &Self::GENOME) -> Self::FIT {
        let mut cppn = Cppn::new(ind.network());
        let mut net_builder = NETBUILDER::new();

        let (behavior, connection_cost, sat) = develop_cppn(&mut cppn,
                                                            &self.substrate_configuration,
                                                            &mut net_builder,
                                                            self.link_expression_range);

        // Evaluate domain specific fitness
        let domain_fitness = self.domain_fitness.fitness(net_builder.network());

        Fitness {
            domain_fitness: domain_fitness,
            behavioral_diversity: 0.0, // will be calculated in `population_metric`
            connection_cost: connection_cost,
            behavior: behavior,
            age_diversity: 0.0, // will be calculated in `population_metric`
            saturation: sat.sum(),
            complexity: ind.complexity(),
        }
    }

    fn mate<R>(&self, rng: &mut R, parent1: &Self::GENOME, parent2: &Self::GENOME) -> Self::GENOME
        where R: Rng
    {
        // XXX set birth_iteration
        self.reproduction.mate(rng, parent1, parent2, 0 /* XXX */)
    }

    fn population_metric(&self, population: &mut RatedPopulation<Self::GENOME, Self::FIT>) {
        PopulationFitness.apply(0 /* XXX */, population);
    }
}

/// Creates a random individual for use by the start generation.
///
/// We start from a minimal topology.
/// If `self.start_connected` is `true`, we add some initial connections.
pub struct RandomGenomeCreator {
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

    /// In case initial random nodes are created, choose activation function from those
    pub start_activation_functions: Vec<GeometricActivationFunction>,

    pub link_weight_range: WeightRange,
}

pub type G = CppnGenome<GeometricActivationFunction>;

impl RandomGenomeCreator {
    pub fn create<R, P>(&self, iteration: usize, rng: &mut R) -> G
        where R: Rng,
              P: Position
    {
        let mut genome = G::new(iteration);

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        // for every dimension we use two inputs e.g. (x1, x2), each for the other end of the
        // connection

        for _d in 0..P::dims() {
            let inp1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
            let inp2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));

            inputs.push(inp1);
            inputs.push(inp2);
        }

        // 1 bias node (constant input of 1.0)
        let bias = genome.add_node(CppnNode::bias(GeometricActivationFunction::Constant1));
        inputs.push(bias);

        // 4 outputs (t,ex,w,r)
        let out_t = genome.add_node(CppnNode::output(GeometricActivationFunction::LinearBipolarClipped));
        let out_ex = genome.add_node(CppnNode::output(GeometricActivationFunction::LinearBipolarClipped));
        let out_w = genome.add_node(CppnNode::output(GeometricActivationFunction::LinearBipolarClipped));
        let out_r = genome.add_node(CppnNode::output(GeometricActivationFunction::LinearBipolarClipped));
        outputs.push(out_t);
        outputs.push(out_ex);
        outputs.push(out_w);
        outputs.push(out_r);

        // make those nodes above immutable for mutation and crossover, as we need them to develop
        // the CPPN.
        genome.protect_nodes();

        for d in 0..P::dims() {
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
            let af = *rng.choose(&self.start_activation_functions).unwrap();
            let _ = genome.add_node(CppnNode::hidden(af));
        }

        genome
    }
}

pub struct Reproduction {
    pub mate_retries: usize,
    pub mating_method_weights: MatingMethodWeights,
    pub mutate_element_prob: Prob,
    pub link_weight_range: WeightRange,
    pub mutate_add_node_random_link_weight: bool,
    pub mutate_drop_node_tournament_k: usize,
    pub mutate_modify_node_tournament_k: usize,
    pub link_weight_creation_sigma: f64,
    pub activation_functions: Vec<GeometricActivationFunction>,
    pub weight_perturbance: WeightPerturbanceMethod,
}

impl Reproduction {
    fn random_hidden_node<R>(&self, rng: &mut R) -> CppnNode<GeometricActivationFunction>
        where R: Rng
    {
        let af = *rng.choose(&self.activation_functions).unwrap();
        CppnNode::hidden(af)
    }

    pub fn mate<R>(&self, rng: &mut R, parent1: &G, parent2: &G, offspring_iteration: usize) -> G
        where R: Rng
    {
        let mut offspring = parent1.fork(offspring_iteration);

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
                MatingMethod::MutateSymmetricJoin => offspring.mutate_symmetric_join(rng),
                MatingMethod::MutateSymmetricFork => offspring.mutate_symmetric_fork(rng),
                MatingMethod::MutateSymmetricConnect => {
                    let link_weight =
                        self.link_weight_range
                            .clip_weight(Weight(gaussian(self.link_weight_creation_sigma, rng)));
                    offspring.mutate_symmetric_connect(link_weight, 5 /* XXX */, rng)
                }

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
}

pub struct Expression {
    pub link_expression_range: (f64, f64),
}

impl Expression {
    pub fn express<NETBUILDER, POS, NT, GRAPH>(&self,
                                               ind: &G,
                                               net_builder: &mut NETBUILDER,
                                               substrate_config: &SubstrateConfiguration<POS, NT>)
                                               -> (Behavior, f64, Saturation)
        where NETBUILDER: NetworkBuilder<POS = POS, NT = NT, Output = GRAPH>,
              POS: Position
    {
        let mut cppn = Cppn::new(ind.network());
        develop_cppn(&mut cppn,
                     substrate_config,
                     net_builder,
                     self.link_expression_range)
    }
}

pub struct PopulationFitness;

impl PopulationFitness {
    pub fn apply(&self, current_iteration: usize, population: &mut RatedPopulation<G, Fitness>) {
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
                let distance = population.fitness(i)
                                         .behavior
                                         .weighted_distance(&population.fitness(j).behavior);
                diversity_i += distance;
                population.fitness_mut(j).behavioral_diversity += distance;
            }

            population.fitness_mut(i).behavioral_diversity = diversity_i;
        }


        // Calculate age diversity

        // calculate average age of population.
        let total_age: usize = population.individuals()
                                         .iter()
                                         .map(|ind| ind.genome().age(current_iteration))
                                         .fold(0, |acc, x| acc+x);
        let avg_age = (total_age as f64) / n as f64;

        // set age_diversity
        for i in 0..n {
            let age = population.individuals()[i].genome().age(current_iteration);
            population.fitness_mut(i).age_diversity = (avg_age - age as f64).abs();
        }
    }
}
