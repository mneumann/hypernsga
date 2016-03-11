use rand::Rng;

#[derive(Debug, Copy, Clone)]
pub struct Prob(f32);

impl Prob {
    pub fn new(p: f32) -> Prob {
        assert!(p >= 0.0 && p <= 1.0);
        Prob(p)
    }

    pub fn flip<R: Rng>(&self, rng: &mut R) -> bool {
        if self.0 < 1.0 {
            let v: f32 = rng.gen(); // half open [0, 1)
            debug_assert!(v >= 0.0 && v < 1.0);
            v < self.0
        } else {
            true
        }
    }
}

/// Takes the fractional part of `num` as probability to
/// round up or down to the next integral value.
pub fn probabilistic_round<R: Rng>(num: f64, rng: &mut R) -> f64 {
    assert!(num >= 0.0);
    let p: f64 = rng.gen(); // half open [0, 1)
    debug_assert!(p >= 0.0 && p < 1.0);
    if p < num.fract() {
        num.trunc() + 1.0
    } else {
        num.trunc()
    }
}
