use hypernsga::network_builder::NetworkBuilder;
use super::Vertex;
use hypernsga::domain_graph::{Neuron};
use hypernsga::substrate::{Position3d, Node};

pub struct VizNetworkBuilder {
    pub point_list: Vec<Vertex>,
    pub link_index_list: Vec<u32>,
}

impl NetworkBuilder for VizNetworkBuilder {
    type POS = Position3d;
    type NT = Neuron;
    type Output = ();

    fn new() -> Self {
        VizNetworkBuilder {
            point_list: Vec::new(),
            link_index_list: Vec::new(),
        }
    }

    fn add_node(&mut self, node: &Node<Self::POS, Self::NT>, _param: f64) {
        assert!(node.index == self.point_list.len());
        let color = match node.node_info {
            Neuron::Input => [0.0, 1.0, 0.0, 1.0],
            Neuron::Hidden => [0.0, 0.0, 0.0, 1.0],
            Neuron::Output => [1.0, 0.0, 0.0, 1.0],
        };
        self.point_list.push(Vertex {
            position: [node.position.x() as f32,
            node.position.y() as f32,
            node.position.z() as f32],
            color: color,
        });
    }

    fn add_link(&mut self,
                source_node: &Node<Self::POS, Self::NT>,
                target_node: &Node<Self::POS, Self::NT>,
                weight1: f64,
                _weight2: f64) {
        let w = weight1.abs();
        debug_assert!(w <= 1.0);

        self.link_index_list.push(source_node.index as u32);
        self.link_index_list.push(target_node.index as u32);
    }

    fn network(self) -> Self::Output {
        ()
    }
}
