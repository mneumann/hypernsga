/// Place `n` points on a unit circle
pub struct Circular2d {
    n: usize,
    i: usize,
}

impl Circular2d {
    pub fn new(n: usize) -> Self {
        Circular2d { n: n, i: 1 }
    }
}

impl Iterator for Circular2d {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i <= self.n {
            let k = self.i as f64;

            let angle_step = 360.0 / self.n as f64;
            let angle = (angle_step * k as f64).to_radians();
            let (x, y) = angle.sin_cos();

            self.i += 1;
            Some((x, y))
        } else {
            None
        }
    }
}
