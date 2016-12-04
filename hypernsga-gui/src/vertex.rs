use std::convert::From;
use hypernsga::substrate::{Node, Position3d};
use hypernsga::domain_graph::Neuron;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

//implement_vertex!(Vertex, position, color);

impl<'a> From<&'a Node<Position3d, Neuron>> for Vertex {
    fn from(node: &'a Node<Position3d, Neuron>) -> Self {
        let color = match node.node_info {
            Neuron::Input => [0.0, 1.0, 0.0, 1.0],
            Neuron::Hidden => [0.0, 0.0, 0.0, 1.0],
            Neuron::Output => [1.0, 0.0, 0.0, 1.0],
        };
        Vertex {
            position: [node.position.x() as f32,
                       node.position.y() as f32,
                       node.position.z() as f32],
            color: color,
        }
    }
}
