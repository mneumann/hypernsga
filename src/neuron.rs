use graph::NodeLabel;
use graph_neighbor_matching::NodeColorWeight;
use std::str::FromStr;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Neuron {
    Input,
    Output,
    Hidden,
}

impl NodeLabel for Neuron {
    fn node_label(&self, _idx: usize) -> Option<String> {
        match *self {
            Neuron::Input => Some("Input".to_owned()),
            Neuron::Hidden => Some("Hidden".to_owned()),
            Neuron::Output => Some("Output".to_owned()),
        }
    }
    fn node_shape(&self) -> &'static str {
        match *self {
            Neuron::Input => "circle",
            Neuron::Hidden => "box",
            Neuron::Output => "doublecircle",
        }
    }
}

impl NodeColorWeight for Neuron {
    fn node_color_weight(&self) -> f32 {
        match *self {
            Neuron::Input => 0.0,
            Neuron::Hidden => 1.0,
            Neuron::Output => 2.0,
        }
    }
}

impl FromStr for Neuron {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "input" => Ok(Neuron::Input),
            "output" => Ok(Neuron::Output),
            "hidden" => Ok(Neuron::Hidden),
            _ => Err("Invalid node type/weight"),
        }
    }
}
