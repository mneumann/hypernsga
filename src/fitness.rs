use nsga2::multi_objective::MultiObjective;
use std::cmp::Ordering;

/// We use a fitness composed of three objectives. Smaller values are "better".

pub struct Fitness {
   /// The domain-specific fitness
   domain_fitness: f32,

   /// The behavioral diversity of the CPPN (negated).
   behavioral_diversity: f32,

   /// The connection cost of the generated network
   connection_cost: f32,
}

impl MultiObjective for Fitness {
   fn num_objectives(&self) -> usize {
      3
   }

   fn cmp_objective(&self, other: &Self, objective: usize) -> Ordering {
      let s = [self.domain_fitness, self.behavioral_diversity, self.connection_cost];
      let o = [other.domain_fitness, other.behavioral_diversity, other.connection_cost];
      s[objective].partial_cmp(&o[objective]).unwrap()
   }

   fn dist_objective(&self, other: &Self, objective: usize) -> f32 {
      let s = [self.domain_fitness, self.behavioral_diversity, self.connection_cost];
      let o = [other.domain_fitness, other.behavioral_diversity, other.connection_cost];
      s[objective] - o[objective]
   }
}
