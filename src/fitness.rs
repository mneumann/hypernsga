use nsga2::multi_objective::MultiObjective;
use nsga2::domination::Domination;
use std::cmp::Ordering;
use behavioral_bitvec::BehavioralBitvec;

/// We use a fitness composed of three objectives. Smaller values are "better".
#[derive(Debug, Clone)]
pub struct Fitness {
    /// The domain-specific fitness (higher value is better!)
    pub domain_fitness: f64,

    /// The behavioral diversity of the CPPN (higher value is better!)
    ///
    /// This value is not normalized towards the number of individuals in the population,
    /// and as such just the sum of all hamming distances.
    pub behavioral_diversity: u64,

    /// The connection cost of the generated network (smaller value is better!)
    pub connection_cost: f64,

    // This is used to determine the behavioral_diversity.
    pub behavioral_bitvec: BehavioralBitvec,

}

impl MultiObjective for Fitness {
    const NUM_OBJECTIVES: usize = 3;

    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        match objective {
            // higher domain_fitness is better!
            0 => self.domain_fitness.partial_cmp(&other.domain_fitness).unwrap().reverse(),
            // higher behavioral_diversity is better!
            1 => self.behavioral_diversity.cmp(&other.behavioral_diversity).reverse(),
            // smaller connection_cost is better
            2 => self.connection_cost.partial_cmp(&other.connection_cost).unwrap(),
            _ => panic!(),
        }
    }

    fn dist_objective(&self, other: &Self, objective: usize) -> f64 {
        match objective {
            0 => self.domain_fitness - other.domain_fitness,
            1 => {
                if self.behavioral_diversity > other.behavioral_diversity {
                    (self.behavioral_diversity - other.behavioral_diversity) as f64
                } else {
                    (other.behavioral_diversity - self.behavioral_diversity) as f64
                }
            }
            2 => self.connection_cost - other.connection_cost,
            _ => panic!(),
        }
    }
}

impl Domination for Fitness {
    fn domination_ord(&self, rhs: &Fitness) -> Ordering {
        let lhs = self;

        let mut left_dom_cnt = 0;
        let mut right_dom_cnt = 0;

        match lhs.domain_fitness.partial_cmp(&rhs.domain_fitness).unwrap() {
            Ordering::Greater => {
                // higher values are better
                left_dom_cnt += 1;
            }
            Ordering::Less => {
                right_dom_cnt += 1;
            }
            Ordering::Equal => {}
        }

        match lhs.behavioral_diversity.cmp(&rhs.behavioral_diversity) {
            Ordering::Greater => {
                // higher values are better
                left_dom_cnt += 1;
            }
            Ordering::Less => {
                right_dom_cnt += 1;
            }
            Ordering::Equal => {}
        }

        match lhs.connection_cost.partial_cmp(&rhs.connection_cost).unwrap() {
            Ordering::Less => {
                // smaller values are better
                left_dom_cnt += 1;
            }
            Ordering::Greater => {
                right_dom_cnt += 1;
            }
            Ordering::Equal => {}
        }

        if left_dom_cnt > 0 && right_dom_cnt == 0 {
            Ordering::Less
        } else if right_dom_cnt > 0 && left_dom_cnt == 0 {
            Ordering::Greater
        } else {
            debug_assert!((left_dom_cnt > 0 && right_dom_cnt > 0) || (left_dom_cnt == 0 && right_dom_cnt == 0));
            Ordering::Equal
        }
    }
}

/// Trait used to evaluate the domain specific fitness

pub trait DomainFitness<G>: Sync where G: Sync {
    fn fitness(&self, g: G) -> f64;
}
