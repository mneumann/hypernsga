use rand::Rng;
use rand::distributions::{WeightedChoice, Weighted, IndependentSample};

#[derive(Debug, Clone, Copy)]
pub enum MatingMethod {
    // Structural Mutation
    MutateAddNode,
    MutateDropNode,
    MutateModifyNode,
    MutateConnect,
    MutateDisconnect,

    MutateSymmetricJoin,
    MutateSymmetricFork,
    MutateSymmetricConnect,

    // Mutation of weights
    MutateWeights,

    // Crossover of weights
    CrossoverWeights,
}

#[derive(Debug, Clone, Copy)]
pub struct MatingMethodWeights {
    pub mutate_add_node: u32,
    pub mutate_drop_node: u32,
    pub mutate_modify_node: u32,
    pub mutate_connect: u32,
    pub mutate_disconnect: u32,
    pub mutate_weights: u32,
    pub mutate_symmetric_join: u32,
    pub mutate_symmetric_fork: u32,
    pub mutate_symmetric_connect: u32,
    pub crossover_weights: u32,
}

impl MatingMethod {
    pub fn random_with<R>(p: &MatingMethodWeights, rng: &mut R) -> MatingMethod
        where R: Rng
    {
        let mut items = [Weighted {
                             weight: p.mutate_add_node,
                             item: MatingMethod::MutateAddNode,
                         },
                         Weighted {
                             weight: p.mutate_drop_node,
                             item: MatingMethod::MutateDropNode,
                         },
                         Weighted {
                             weight: p.mutate_modify_node,
                             item: MatingMethod::MutateModifyNode,
                         },
                         Weighted {
                             weight: p.mutate_connect,
                             item: MatingMethod::MutateConnect,
                         },
                         Weighted {
                             weight: p.mutate_disconnect,
                             item: MatingMethod::MutateDisconnect,
                         },

                         Weighted {
                             weight: p.mutate_symmetric_join,
                             item: MatingMethod::MutateSymmetricJoin,
                         },
                         Weighted {
                             weight: p.mutate_symmetric_fork,
                             item: MatingMethod::MutateSymmetricFork,
                         },
                         Weighted {
                             weight: p.mutate_symmetric_connect,
                             item: MatingMethod::MutateSymmetricConnect,
                         },

                         Weighted {
                             weight: p.mutate_weights,
                             item: MatingMethod::MutateWeights,
                         },
                         Weighted {
                             weight: p.crossover_weights,
                             item: MatingMethod::CrossoverWeights,
                         }];
        WeightedChoice::new(&mut items).ind_sample(rng)
    }
}
