use std::f64::consts::PI;

/// Distributes `n` points roughly within a circle using sunflower seed placement [1,2].
///
/// [1]: http://demonstrations.wolfram.com/SunflowerSeedArrangements/
/// [2]: http://www.sciencedirect.com/science/article/pii/0025556479900804
pub struct SunflowerSeed2d {
    n: usize,
    i: usize,
    b: f64,
    phi2: f64,
}

impl SunflowerSeed2d {
    pub fn new(n: usize, alpha: f64) -> Self {
        SunflowerSeed2d {
            n: n,
            i: 1,
            b: (alpha * (n as f64).sqrt()).round(),
            phi2: (((5.0f64).sqrt() + 1.0) / 2.0).powi(2),
        }
    }
}

fn radius(k: f64, n: f64, b: f64) -> f64 {
    if k > n - b {
        1.0
    } else {
        (k - 0.5).sqrt() / (n - (b + 1.0) / 2.0).sqrt()
    }
}

impl Iterator for SunflowerSeed2d {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i <= self.n {
            let k = self.i as f64;
            self.i += 1;

            let r = radius(k, self.n as f64, self.b);
            let theta = (2.0 * PI * k) / self.phi2;
            let (sin, cos) = theta.sin_cos();
            let x = r * cos;
            let y = r * sin;
            Some((x, y))
        } else {
            None
        }
    }
}

#[test]
fn test_sunflower() {
    let eps = 0.0001;
    let mut p = SunflowerSeed2d::new(1, 0.0);
    let (ex, ey) = (-0.7373688780783197, 0.6754902942615238); // expected
    let (x, y) = p.next().unwrap();
    assert!((x - ex).abs() < eps);
    assert!((y - ey).abs() < eps);

    assert_eq!(None, p.next());
}

#[test]
fn test_sunflower2() {
    let eps = 0.0001;
    let expected: Vec<(f64, f64)> = vec![(-0.2457896260261066, 0.225163431420508),
                                         (0.0504752656994349, -0.575139618602218),
                                         (0.6084388609788619, 0.7936007512916965),
                                         (-0.9847134853154287, -0.17418195037931164),
                                         (0.8437552948123972, -0.5367280526263227),
                                         (-0.2596043049014903, 0.9657150743757779),
                                         (-0.4609070247133692, -0.8874484292452546),
                                         (0.9393212963241181, 0.343038630874102)];
    let values: Vec<_> = SunflowerSeed2d::new(8, 2.0).collect();
    assert_eq!(expected.len(), values.len());

    for (&(ex, ey), &(x, y)) in expected.iter().zip(values.iter()) {
        assert!((x - ex).abs() < eps);
        assert!((y - ey).abs() < eps);
    }
}
