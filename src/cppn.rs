use cppn_ext::cppn::{Cppn, CppnNode};
use cppn_ext::position::Position;
use cppn_ext::activation_function::ActivationFunction;
use weight::Weight;
use substrate::{Substrate, Node};
use behavioral_bitvec::BehavioralBitvec;

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
