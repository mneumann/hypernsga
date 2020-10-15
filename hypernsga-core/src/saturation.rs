pub struct Saturation {
    pub zero: f64,
    pub low: f64,
    pub high: f64,
}

impl Saturation {
    pub fn sum(&self) -> f64 {
        self.zero + self.low + self.high
    }
}
