use cppn_ext::cppn::{Cppn, CppnNode};
use cppn_ext::activation_function::{GeometricActivationFunction, ActivationFunction};
use weight::{Weight, WeightRange, WeightPerturbanceMethod};
use substrate::{Position, SubstrateConfiguration};
use behavioral_bitvec::BehavioralBitvec;
use genome::Genome;
use nsga2::driver::Driver;
use nsga2::population::RatedPopulation;
use fitness::{Fitness, DomainFitness};
use rand::Rng;
use mating::{MatingMethod, MatingMethodWeights};
use prob::Prob;
use network_builder::NetworkBuilder;
use std::marker::PhantomData;

pub type CppnGenome<AF> where AF: ActivationFunction = Genome<CppnNode<AF>>;

const CPPN_OUTPUT_LINK_WEIGHT1: usize = 0;
const CPPN_OUTPUT_LINK_WEIGHT2: usize = 1;
const CPPN_OUTPUT_LINK_EXPRESSION: usize = 2;
const CPPN_OUTPUT_NODE_WEIGHT: usize = 3;

/// Develops a network out of the CPPN
///
/// Returns the BehavioralBitvec and Connection Cost of the developed network

fn develop_cppn<'a, P, AF, T, V>(cppn: &mut Cppn<CppnNode<AF>, Weight, ()>,
                                 substrate_config: &SubstrateConfiguration<'a, P, T>,
                                 visitor: &mut V,
                                 leo_threshold: f64)
                                 -> (BehavioralBitvec, f64)
    where P: Position,
          AF: ActivationFunction,
          V: NetworkBuilder<POS = P, NT = T>
{
    // our CPPN has four outputs: link weight 1, link weight 2, link expression output, node weight
    assert!(cppn.output_count() == 4);
    assert!(cppn.input_count() == 6);

    let nodes = substrate_config.nodes();
    let links = substrate_config.links();
    let null_position = substrate_config.null_position();

    let mut bitvec = BehavioralBitvec::new(4 * (nodes.len() + links.len()));
    let mut connection_cost = 0.0;

    // First visit all nodes

    for node in nodes.iter() {
        let inputs = [node.position.coords(), null_position.coords()];
        cppn.process(&inputs[..]);

        let link_weight1 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT1).unwrap();
        let link_weight2 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT2).unwrap();
        let link_expression = cppn.read_output(CPPN_OUTPUT_LINK_EXPRESSION).unwrap();
        let node_weight = cppn.read_output(CPPN_OUTPUT_NODE_WEIGHT).unwrap();

        bitvec.push(link_weight1);
        bitvec.push(link_weight2);
        bitvec.push(link_expression);
        bitvec.push(node_weight);

        visitor.add_node(node, node_weight)
    }

    for &(source_node, target_node) in links.iter() {
        let inputs = [source_node.position.coords(), target_node.position.coords()];
        cppn.process(&inputs[..]);

        let link_weight1 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT1).unwrap();
        let link_weight2 = cppn.read_output(CPPN_OUTPUT_LINK_WEIGHT2).unwrap();
        let link_expression = cppn.read_output(CPPN_OUTPUT_LINK_EXPRESSION).unwrap();
        let node_weight = cppn.read_output(CPPN_OUTPUT_NODE_WEIGHT).unwrap();

        bitvec.push(link_weight1);
        bitvec.push(link_weight2);
        bitvec.push(link_expression);
        bitvec.push(node_weight);

        if link_expression > leo_threshold {
            let distance_sq = source_node.position.distance_square(&target_node.position);
            debug_assert!(distance_sq >= 0.0);
            connection_cost += distance_sq;
            visitor.add_link(source_node, target_node, link_weight1, link_weight2);
        }
    }

    return (bitvec, connection_cost);
}

/// Determines the domain fitness of `G`

pub struct CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G> + 'a,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync,
          G: Sync,
{
    mating_method_weights: MatingMethodWeights,
    activation_functions: Vec<GeometricActivationFunction>,
    mutate_element_prob: Prob,
    weight_perturbance: WeightPerturbanceMethod,
    link_weight_range: WeightRange,

    mutate_add_node_random_link_weight: bool,
    mutate_drop_node_tournament_k: usize,
    mutate_modify_node_tournament_k: usize,

    mate_retries: usize,

    link_expression_threshold: f64,

    substrate_configuration: SubstrateConfiguration<'a, P, T>,
    domain_fitness: &'a DOMFIT,
    _phantom: PhantomData<NETBUILDER>,
}

impl<'a, DOMFIT, G, P, T, NETBUILDER> CppnDriver<'a, DOMFIT, G, P, T, NETBUILDER>
    where DOMFIT: DomainFitness<G>,
          P: Position + Sync + 'a,
          T: Sync + 'a,
          NETBUILDER: NetworkBuilder<POS = P, NT = T, Output = G> + Sync,
          G: Sync,
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
          G: Sync,
{
    type IND = CppnGenome<GeometricActivationFunction>;
    type FIT = Fitness;

    /// Creates a random individual for use by the start generation.
    ///
    /// We start from a minimal topology.

    fn random_individual<R>(&self, _rng: &mut R) -> Self::IND
        where R: Rng
    {
        let mut genome = Self::IND::new();

        // 6 inputs (x1,y1,z1, x2,y2,z2)
        let _inp_x1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let _inp_y1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let _inp_z1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let _inp_x2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let _inp_y2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let _inp_z2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));

        // 1 bias node (constant input of 1.0)
        let _bias = genome.add_node(CppnNode::bias(GeometricActivationFunction::Constant1));

        // 4 outputs (t,w,ex,r)
        let _out_t = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        let _out_w = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        let _out_ex = genome.add_node(CppnNode::output(GeometricActivationFunction::Linear));
        let _out_r = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));

        // make those nodes above immutable for mutation and crossover, as we need them to
        // develop the CPPN.
        genome.protect_nodes();

        // XXX: Add x-distance and y-distance Gaussian seed nodes.
        // XXX: Add initial random connections?

        genome
    }

    fn fitness(&self, ind: &Self::IND) -> Self::FIT {
        let mut cppn = Cppn::new(ind.network());
        let mut net_builder = NETBUILDER::new();

        let (behavioral_bitvec, connection_cost) = develop_cppn(&mut cppn,
                                                                &self.substrate_configuration,
                                                                &mut net_builder,
                                                                self.link_expression_threshold);

        // Evaluate domain specific fitness
        let domain_fitness = self.domain_fitness.fitness(net_builder.network());

        Fitness {
            domain_fitness: domain_fitness,
            behavioral_diversity: 0, // will be calculated in `population_metric`
            connection_cost: connection_cost,
            behavioral_bitvec: behavioral_bitvec,
        }
    }

    fn mate<R>(&self, rng: &mut R, parent1: &Self::IND, parent2: &Self::IND) -> Self::IND
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
                    let link_weight = self.link_weight_range.random_weight(rng);
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

    fn population_metric(&self, population: &mut RatedPopulation<Self::IND, Self::FIT>) {
        // Determine the behavioral_diversity as average hamming distance to all other individuals.
        // hamming distance is symmetric.
        let n = population.len();
        for i in 0..n {
            // determine  hehavioral diversity for `i`.
            let mut diversity_i = 0;

            // XXX: parallelize this loop
            for j in i + 1..n {
                let distance = population.fitness()[i]
                                   .behavioral_bitvec
                                   .hamming_distance(&population.fitness()[j].behavioral_bitvec);
                diversity_i += distance;
                population.fitness_mut()[j].behavioral_diversity += distance;
            }

            population.fitness_mut()[i].behavioral_diversity = diversity_i;
        }
    }
}
