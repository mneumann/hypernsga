use ::network_builder::NetworkBuilder;
use ::domain_graph::Neuron;
use ::substrate::{Position3d, Node};

// For vizualizing a network (e.g. converting into a 3d mesh)
pub struct VizNetworkBuilder<T> {
    pub point_list: Vec<T>,
    pub link_index_list: Vec<u32>,
}

pub trait FromRef<T> {
    fn from_ref(t: &T) -> Self;
}

impl<'a, V> NetworkBuilder for VizNetworkBuilder<V>
where V: FromRef<Node<Position3d, Neuron>> {

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
        self.point_list.push(FromRef::from_ref(node));
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
