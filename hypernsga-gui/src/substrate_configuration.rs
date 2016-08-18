use hypernsga::domain_graph::{Neuron, NodeCount};
use hypernsga::substrate::{NodeSet, Substrate, SubstrateConfiguration, Position3d};
use hypernsga::distribute::DistributeInterval;

pub fn substrate_configuration(node_count: &NodeCount)
                               -> SubstrateConfiguration<Position3d, Neuron> {
    let mut substrate: Substrate<Position3d, Neuron> = Substrate::new();

    let input_nodeset = NodeSet::single(0);
    let hidden_nodeset = NodeSet::single(1);
    let output_nodeset = NodeSet::single(2);

    let nodeset_links = &[(input_nodeset, hidden_nodeset),
                          (hidden_nodeset, output_nodeset),
                          (input_nodeset, output_nodeset) /* (output_nodeset, input_nodeset) */];

    // let nodeset_links = &[(input_nodeset, hidden_nodeset), (hidden_nodeset, output_nodeset),
    // (output_nodeset, input_nodeset)
    // ];


    // Input layer
    {
        let z = -1.0;
        // let z = 0.0;
        // let y = 0.0;
        for x in DistributeInterval::new(node_count.inputs,
                                         -1.0 * node_count.inputs as f64,
                                         1.0 * node_count.inputs as f64) {
            // for x in DistributeInterval::new(node_count.inputs, -1.0, 1.0) {
            let y = 0.0; //0.1 * (1.0 - x.powi(8));
            substrate.add_node(Position3d::new(x, y, z), Neuron::Input, input_nodeset);
        }
    }

    // Hidden
    {
        let z = 0.0;
        // let y = 0.0;
        for x in DistributeInterval::new(node_count.hidden,
                                         -1.0 * node_count.hidden as f64,
                                         1.0 * node_count.hidden as f64) {
            // for x in DistributeInterval::new(node_count.hidden, -1.0, 1.0) {
            // let y = (1.0 - x.powi(8));
            let y = 0.0; //-1.0;
            substrate.add_node(Position3d::new(x, y, z), Neuron::Hidden, hidden_nodeset);
        }
    }

    // Outputs
    {
        let z = 1.0;
        // let z = 1.0;
        // let y = 0.0;
        // let mut z = DistributeInterval::new(node_count.outputs, -0.1, 0.1);
        for x in DistributeInterval::new(node_count.outputs,
                                         -1.0 * node_count.outputs as f64,
                                         1.0 * node_count.outputs as f64) {
            // for x in DistributeInterval::new(node_count.outputs, -1.0, 1.0) {
            // let y = -0.1 * (1.0 - x.powi(8));
            let y = 0.0;
            // substrate.add_node(Position3d::new(x, y, -z.next().unwrap()),
            substrate.add_node(Position3d::new(x, y, z), Neuron::Output, output_nodeset);
        }
    }

    return substrate.to_configuration(nodeset_links);
}
