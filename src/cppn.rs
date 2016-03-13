use cppn_ext::cppn::{Cppn, CppnNode};
use cppn_ext::position::Position;
use cppn_ext::activation_function::{GeometricActivationFunction, ActivationFunction};
use weight::{Weight, WeightRange, WeightPerturbanceMethod};
use substrate::{Substrate, Node};
use behavioral_bitvec::BehavioralBitvec;
use genome::Genome;
use nsga2::driver::Driver;
use nsga2::population::{UnratedPopulation, RatedPopulation};
use fitness::Fitness;
use rand::Rng;
use mating::{MatingMethod, MatingMethodWeights};
use prob::Prob;

pub type CppnGenome<AF> where AF: ActivationFunction = Genome<CppnNode<AF>>;

pub trait NetworkBuilderVisitor<P, T> where P: Position {
    fn add_node(&mut self, node: &Node<P, T>, param: f64);
    fn add_link(&mut self,
                source_node: &Node<P, T>,
                target_node: &Node<P, T>,
                weight1: f64,
                weight2: f64);
}

const CPPN_OUTPUT_LINK_WEIGHT1: usize = 0;
const CPPN_OUTPUT_LINK_WEIGHT2: usize = 1;
const CPPN_OUTPUT_LINK_EXPRESSION: usize = 2;
const CPPN_OUTPUT_NODE_WEIGHT: usize = 3;

/// Develops a network out of the CPPN
///
/// Returns the BehavioralBitvec and Connection Cost of the developed network

fn develop_cppn<P, AF, T, V>(cppn: &mut Cppn<CppnNode<AF>, Weight, ()>,
                             null_position: &P,
                             nodes: &[Node<P, T>],
                             links: &[(&Node<P, T>, &Node<P, T>)],
                             visitor: &mut V,
                             leo_threshold: f64)
                             -> (BehavioralBitvec, f64)
    where P: Position,
          AF: ActivationFunction,
          V: NetworkBuilderVisitor<P, T>
{

    // our CPPN has four outputs: link weight 1, link weight 2, link expression output, node weight
    assert!(cppn.output_count() == 4);
    assert!(cppn.input_count() == 6);

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

pub struct CppnDriver {
    mating_method_weights: MatingMethodWeights,
    activation_functions: Vec<GeometricActivationFunction>,
    mutate_element_prob: Prob,
    weight_perturbance: WeightPerturbanceMethod,
    link_weight_range: WeightRange,

    mutate_add_node_random_link_weight: bool,
    mutate_drop_node_tournament_k: usize,
    mutate_modify_node_tournament_k: usize,
}

impl CppnDriver {
    fn random_hidden_node<R>(&self, rng: &mut R) -> CppnNode<GeometricActivationFunction> where R: Rng {
        let af = *rng.choose(&self.activation_functions).unwrap();
        CppnNode::hidden(af)
    }
}

impl Driver for CppnDriver {
    type IND = CppnGenome<GeometricActivationFunction>;
    type FIT = Fitness;

    /// Creates a random individual for use by the start generation.
    /// 
    /// We start from a minimal topology.

    fn random_individual<R>(&self, rng: &mut R) -> Self::IND where R: Rng {
        let mut genome = Self::IND::new(); 

        // 6 inputs (x1,y1,z1, x2,y2,z2)
        let inp_x1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let inp_y1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let inp_z1 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let inp_x2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let inp_y2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));
        let inp_z2 = genome.add_node(CppnNode::input(GeometricActivationFunction::Linear));

        // 1 bias node (constant input of 1.0)
        let bias = genome.add_node(CppnNode::bias(GeometricActivationFunction::Constant1));

        // 4 outputs (t,w,ex,r)
        let out_t  = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        let out_w  = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));
        let out_ex = genome.add_node(CppnNode::output(GeometricActivationFunction::Linear));
        let out_r  = genome.add_node(CppnNode::output(GeometricActivationFunction::BipolarGaussian));

        // make those nodes above immutable for mutation and crossover, as we need them to
        // develop the CPPN.
        genome.protect_nodes();

        // XXX: Add x-distance and y-distance Gaussian seed nodes.
        // XXX: Add initial random connections?

        genome
    }

    fn fitness(&self, ind: &Self::IND) -> Self::FIT {
        Fitness {
            domain_fitness: 0.0,
            behavioral_diversity: 0.0,
            connection_cost: 0.0,
        }
    }

    fn mate<R>(&self, rng: &mut R, parent1: &Self::IND, parent2: &Self::IND) -> Self::IND where R: Rng {
        let mut offspring = parent1.clone();

        match MatingMethod::random_with(&self.mating_method_weights, rng) {
            MatingMethod::MutateAddNode => {
                let link_weight =  if self.mutate_add_node_random_link_weight {
                    Some(self.link_weight_range.random_weight(rng))
                } else {
                    // duplicate existing node weight
                    None
                };
                let hidden_node = self.random_hidden_node(rng);
                let _modified = offspring.mutate_add_node(hidden_node, link_weight, rng);
            }
            MatingMethod::MutateDropNode => {
                let _modified = offspring.mutate_drop_node(self.mutate_drop_node_tournament_k, rng);
            }
            MatingMethod::MutateModifyNode => {
                let hidden_node = self.random_hidden_node(rng);
                let _modified = offspring.mutate_modify_node(hidden_node, self.mutate_modify_node_tournament_k, rng);
            }
            MatingMethod::MutateConnect => {
                let link_weight = self.link_weight_range.random_weight(rng);
                let _modified = offspring.mutate_connect(link_weight, rng);
            }
            MatingMethod::MutateDisconnect => {
                let _modified = offspring.mutate_disconnect(rng);
            }
            MatingMethod::MutateWeights => {
                let _modifications = offspring.mutate_weights(self.mutate_element_prob,
                                                         &self.weight_perturbance,
                                                         &self.link_weight_range,
                                                         rng);
            }
            MatingMethod::CrossoverWeights => {
                let _modifications = offspring.crossover_weights(parent2, rng);
            }
        }

        return offspring;
    }

    fn population_metric(&self, population: &mut RatedPopulation<Self::IND, Self::FIT>) {
    }
}
