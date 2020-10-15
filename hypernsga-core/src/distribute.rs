/// Distribute n points equally within the interval [left, right]

pub struct DistributeInterval {
    n: usize,
    i: usize,
    left: f64,
    right: f64,
}

impl DistributeInterval {
    pub fn new(n: usize, left: f64, right: f64) -> Self {
        assert!(left <= right);
        DistributeInterval {
            n: n,
            i: 0,
            left: left,
            right: right,
        }
    }
}

impl Iterator for DistributeInterval {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }

        debug_assert!(self.n > 0);

        let width = self.right - self.left;
        let step = width / self.n as f64;
        let start = self.left + (step / 2.0);

        let new = start + (self.i as f64) * step;
        self.i += 1;

        debug_assert!(new >= self.left && new <= self.right);

        return Some(new);
    }
}

#[test]
fn test_distribute_interval() {
    let mut iter = DistributeInterval::new(0, -1.0, 1.0);
    assert_eq!(None, iter.next());

    let mut iter = DistributeInterval::new(1, -1.0, 1.0);
    assert_eq!(Some(0.0), iter.next());
    assert_eq!(None, iter.next());

    let mut iter = DistributeInterval::new(3, -1.0, 1.0);
    assert_eq!(-66, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(0, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(66, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(None, iter.next());

    let mut iter = DistributeInterval::new(4, -1.0, 1.0);
    assert_eq!(-75, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(-25, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(25, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(75, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(None, iter.next());

    let mut iter = DistributeInterval::new(5, -1.0, 1.0);
    assert_eq!(-80, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(-40, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(0, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(40, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(80, (iter.next().unwrap() * 100.0) as isize);
    assert_eq!(None, iter.next());

    let mut iter = DistributeInterval::new(3, 0.0, 8.0);
    assert_eq!(1, iter.next().unwrap() as usize);
    assert_eq!(4, iter.next().unwrap() as usize);
    assert_eq!(6, iter.next().unwrap() as usize);
    assert_eq!(None, iter.next());

    let mut iter = DistributeInterval::new(3, 0.0, 9.0);
    assert_eq!(1, iter.next().unwrap() as usize);
    assert_eq!(4, iter.next().unwrap() as usize);
    assert_eq!(7, iter.next().unwrap() as usize);
    assert_eq!(None, iter.next());
}
