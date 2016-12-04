use ::domain_graph::Neuron;
use ::substrate::{SubstrateConfiguration, Position};
use ::cppn::{CppnNodeKind, Expression, G, ActivationFunction};
use ::network_builder::NetworkBuilder;
use ::network_builder::gml::GMLNetworkBuilder;
use ::network_builder::dot::DotNetworkBuilder;

use std::io::Write;
use std::fs::File;

pub fn write_net_gml<P>(filename: &str,
                expression: &Expression,
                genome: &G,
                substrate_config: &SubstrateConfiguration<P, Neuron>)
    where P: Position
{
    let mut file = File::create(filename).unwrap();
    let mut network_builder = GMLNetworkBuilder::new();
    network_builder.set_writer(&mut file);
    network_builder.begin();
    let (_behavior, _connection_cost, _) = expression.express(genome, &mut network_builder, substrate_config);
    network_builder.end();
}

pub fn write_net_dot<P>(filename: &str,
                expression: &Expression,
                genome: &G,
                substrate_config: &SubstrateConfiguration<P, Neuron>)
    where P: Position
{
    let mut file = File::create(filename).unwrap();
    let mut network_builder = DotNetworkBuilder::new();
    network_builder.set_writer(&mut file);
    network_builder.begin();
    let (_behavior, _connection_cost, _) = expression.express(genome, &mut network_builder, substrate_config);
    network_builder.end();
}

pub fn write_cppn_dot(filename: &str, genome: &G)
{
    let mut file = File::create(filename).unwrap();
    let network = genome.network();

    writeln!(&mut file,
             "digraph {{
graph [
  layout=neato,
  rankdir = \"TB\",
  \
  overlap=false,
  compound = true,
  nodesep = 1,
  ranksep = \
  2.0,
  splines = \"polyline\",
];
node [fontname = Helvetica];
")
        .unwrap();

    network.each_node_with_index(|node, node_idx| {
        let s = match node.node_type().kind {
            CppnNodeKind::Input => {
                let label = match node_idx.index() {
                    0 => "x1",
                    1 => "y1",
                    2 => "z1",
                    3 => "x2",
                    4 => "y2",
                    5 => "z2",
                    _ => "X",
                };

                // XXX label
                format!("shape=egg,label={},rank=min,style=filled,color=grey",
                        label)
            }
            CppnNodeKind::Bias => {
                assert!(node_idx.index() == 6);
                format!("shape=egg,label=1.0,rank=min,style=filled,color=grey")
            }
            CppnNodeKind::Output => {
                let label = match node_idx.index() {
                    7 => "t",
                    8 => "ex",
                    9 => "w",
                    10 => "r", 
                    _ => panic!(),
                };
                format!("shape=doublecircle,label={},rank=max,style=filled,\
                                             fillcolor=yellow,color=grey",
                                             label)
            }
            CppnNodeKind::Hidden => {
                format!("shape=box,label={}",
                        node.node_type().activation_function.name())
            }
        };
        writeln!(&mut file, "{} [{}];", node_idx.index(), s).unwrap();
    });
    network.each_link_ref(|link_ref| {
        let w = link_ref.link().weight().0;
        let color = if w >= 0.0 { "black" } else { "red" };
        writeln!(&mut file,
                 "{} -> {} [color={}];",
                 link_ref.link().source_node_index().index(),
                 link_ref.link().target_node_index().index(),
                 color)
            .unwrap();
    });

    writeln!(&mut file, "}}").unwrap();
}
