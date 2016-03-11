use rand::Rng;
use rand::distributions::{WeightedChoice, Weighted, IndependentSample};

// XXX: Unify Crossover and Mutate.

#[derive(Debug, Clone, Copy)]
pub enum MutateMethod {
    ModifyWeight,
    AddConnection,
    EnableConnection,
    DeleteConnection,
    AddNode,
}

#[derive(Debug, Clone, Copy)]
pub struct MutateMethodWeighting {
    pub w_modify_weight: u32,
    pub w_add_connection: u32,
    pub w_enable_connection: u32,
    pub w_delete_connection: u32,
    pub w_add_node: u32,
}

impl MutateMethod {
    pub fn random_with<R: Rng>(p: &MutateMethodWeighting, rng: &mut R) -> MutateMethod {
        let mut items = [Weighted {
                             weight: p.w_modify_weight,
                             item: MutateMethod::ModifyWeight,
                         },
                         Weighted {
                             weight: p.w_add_connection,
                             item: MutateMethod::AddConnection,
                         },
                         Weighted {
                             weight: p.w_enable_connection,
                             item: MutateMethod::EnableConnection,
                         },
                         Weighted {
                             weight: p.w_delete_connection,
                             item: MutateMethod::DeleteConnection,
                         },
                         Weighted {
                             weight: p.w_add_node,
                             item: MutateMethod::AddNode,
                         }];
        WeightedChoice::new(&mut items).ind_sample(rng)
    }
}
