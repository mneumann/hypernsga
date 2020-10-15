use rand::distributions::{IndependentSample, Normal};
use rand::{Closed01, Rng};

/// Represents a connection weight.
#[derive(Debug, Clone, Copy)]
pub struct Weight(pub f64);

impl Weight {
    pub fn inv(self) -> Self {
        Weight(-self.0)
    }
}

impl Into<f64> for Weight {
    fn into(self) -> f64 {
        self.0
    }
}

impl Into<f32> for Weight {
    fn into(self) -> f32 {
        self.0 as f32
    }
}

/// Represents the range of a connection weight. The range is closed,
/// i.e. including both endpoints [low, high].
#[derive(Debug, Clone, Copy)]
pub struct WeightRange {
    high: f64,
    low: f64,
}

impl WeightRange {
    pub fn new(high: f64, low: f64) -> WeightRange {
        assert!(high >= low);
        WeightRange {
            high: high,
            low: low,
        }
    }

    pub fn high(&self) -> Weight {
        Weight(self.high)
    }

    pub fn low(&self) -> Weight {
        Weight(self.low)
    }

    pub fn unipolar(magnitude: f64) -> WeightRange {
        if magnitude >= 0.0 {
            WeightRange {
                high: magnitude,
                low: 0.0,
            }
        } else {
            WeightRange {
                high: 0.0,
                low: magnitude,
            }
        }
    }

    pub fn bipolar(magnitude: f64) -> WeightRange {
        assert!(magnitude >= 0.0);
        WeightRange {
            high: magnitude,
            low: -magnitude,
        }
    }

    pub fn in_range(&self, weight: Weight) -> bool {
        weight.0 >= self.low && weight.0 <= self.high
    }

    pub fn random_weight<R: Rng>(&self, rng: &mut R) -> Weight {
        let w = rng.gen::<Closed01<f64>>().0;
        debug_assert!(w >= 0.0 && w <= 1.0);

        let weight = Weight(((self.high - self.low) * w) + self.low);

        debug_assert!(self.in_range(weight));

        weight
    }

    pub fn clip_weight(&self, weight: Weight) -> Weight {
        let clipped = if weight.0 >= self.high {
            Weight(self.high)
        } else if weight.0 <= self.low {
            Weight(self.low)
        } else {
            weight
        };

        debug_assert!(self.in_range(clipped));

        clipped
    }
}

/// Defines a perturbance method.
#[derive(Debug, Clone, Copy)]
pub enum WeightPerturbanceMethod {
    JiggleUniform { range: WeightRange },
    JiggleGaussian { sigma: f64 },
    Random,
}

pub fn gaussian<R: Rng>(sigma: f64, rng: &mut R) -> f64 {
    let normal = Normal::new(0.0, sigma);
    normal.ind_sample(rng)
}

impl WeightPerturbanceMethod {
    pub fn perturb<R: Rng>(
        &self,
        weight: Weight,
        weight_range: &WeightRange,
        rng: &mut R,
    ) -> Weight {
        match *self {
            WeightPerturbanceMethod::Random => weight_range.random_weight(rng),
            WeightPerturbanceMethod::JiggleUniform { range } => {
                weight_range.clip_weight(Weight(weight.0 + range.random_weight(rng).0))
            }
            WeightPerturbanceMethod::JiggleGaussian { sigma } => {
                weight_range.clip_weight(Weight(weight.0 + gaussian(sigma, rng)))
            }
        }
    }
}
