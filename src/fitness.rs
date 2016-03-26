use nsga2::multi_objective::MultiObjective;
use nsga2::domination::Domination;
use std::cmp::Ordering;
use behavioral_bitvec::BehavioralBitvec;

#[derive(Clone, Debug)]
pub struct Behavior {
    pub bv_link_weight1: BehavioralBitvec,
    pub bv_link_weight2: BehavioralBitvec,
    pub bv_link_expression: BehavioralBitvec,
    pub bv_node_weight: BehavioralBitvec,
}

impl Behavior {
    pub fn weighted_distance(&self, other: &Self) -> f64 {
        let d1 = self.bv_link_weight1.hamming_distance(&other.bv_link_weight1) as f64;
        let d2 = self.bv_link_expression.hamming_distance(&other.bv_link_expression) as f64;
        d1 + d2
    }
}

/// We use a fitness composed of three objectives. Smaller values are "better".
#[derive(Debug, Clone)]
pub struct Fitness {
    /// The domain-specific fitness (higher value is better!)
    pub domain_fitness: f64,

    /// The behavioral diversity of the CPPN (higher value is better!)
    ///
    /// This value is not normalized towards the number of individuals in the population,
    /// and as such is just the sum of all hamming distances.
    pub behavioral_diversity: f64,

    /// The connection cost of the generated network (smaller value is better!)
    pub connection_cost: f64,

    // each genome records 
    pub age_diversity: f64,

    pub saturation: f64,

    pub complexity: f64,

    // This is used to determine the behavioral_diversity.
    pub behavior: Behavior,
}

impl MultiObjective for Fitness {
    fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
        match objective {
            // higher domain_fitness is better!
            0 => self.domain_fitness.partial_cmp(&other.domain_fitness).unwrap().reverse(),
            // higher behavioral_diversity is better!
            1 => self.behavioral_diversity.partial_cmp(&other.behavioral_diversity).unwrap().reverse(),
            // smaller connection_cost is better
            2 => self.connection_cost.partial_cmp(&other.connection_cost).unwrap(),
            // higer age_diversity is better
            3 => self.age_diversity.partial_cmp(&other.age_diversity).unwrap().reverse(),
            // lower saturation is better
            4 => self.saturation.partial_cmp(&other.saturation).unwrap(),
            // lower complexity is better
            5 => self.complexity.partial_cmp(&other.complexity).unwrap(),

            _ => panic!(),
        }
    }

    fn dist_objective(&self, other: &Self, objective: usize) -> f64 {
        match objective {
            0 => self.domain_fitness - other.domain_fitness,
            1 => {
                self.behavioral_diversity - other.behavioral_diversity
            }
            2 => self.connection_cost - other.connection_cost,
            3 => self.age_diversity - other.age_diversity,
            4 => self.saturation - other.saturation,
            5 => self.complexity - other.complexity,
            _ => panic!(),
        }
    }
}

impl Domination for Fitness {
    fn domination_ord(&self, rhs: &Fitness, objectives: &[usize]) -> Ordering {
        let lhs = self;

        let mut left_dom_cnt = 0;
        let mut right_dom_cnt = 0;

        for &i in objectives.iter() {
            match lhs.cmp_objective(rhs, i) {
                Ordering::Less => {
                    left_dom_cnt += 1;
                }
                Ordering::Greater => {
                    right_dom_cnt += 1;
                }
                Ordering::Equal => {}
            }
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
