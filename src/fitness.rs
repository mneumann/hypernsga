use std::cmp::Ord;
use std::cmp::Ordering;
use std::ops::{Add, Div};
use std::num::Zero;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Fitness(f64);

impl Fitness {
    pub fn new(fitness: f64) -> Fitness {
        assert!(fitness >= 0.0);
        Fitness(fitness)
    }

    pub fn get(&self) -> f64 {
        self.0
    }
}

impl Eq for Fitness {}

impl Zero for Fitness {
    fn zero() -> Self {
        Fitness(0.0)
    }
}

impl Add for Fitness {
    type Output = Fitness;
    fn add(self, rhs: Self) -> Self::Output {
        Fitness(self.0 + rhs.0)
    }
}

impl Div for Fitness {
    type Output = Fitness;
    fn div(self, rhs: Self) -> Self::Output {
        Fitness(self.0 / rhs.0)
    }
}

impl Ord for Fitness {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Less)
    }
}
