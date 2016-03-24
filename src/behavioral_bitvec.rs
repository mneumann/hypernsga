use primal_bit::BitVec;
use hamming;

#[derive(Clone, Debug)]
pub struct BehavioralBitvec {
    bitvec: BitVec,
    bitpos: usize,
}

impl BehavioralBitvec {
    pub fn new(n: usize) -> Self {
        BehavioralBitvec {
            bitvec: BitVec::from_elem(n, false),
            bitpos: 0,
        }
    }

    pub fn hamming_distance(&self, other: &Self) -> u64 {
        hamming::distance_fast(self.bitvec.as_bytes(), other.bitvec.as_bytes()).unwrap()
    }

    #[inline]
    pub fn push_bit(&mut self, bit: bool) {
        if bit {
            self.bitvec.set(self.bitpos, true);
        }
        self.bitpos += 1;
    }

    #[inline]
    pub fn push(&mut self, output: f64) {
        self.push_bit(output >= 0.0)
    }
}

#[test]
fn test_behavioral_bitvec() {
    let mut a = BehavioralBitvec::new(3);
    let mut b = BehavioralBitvec::new(3);

    // [1, 1, 0]
    a.push(1.0);
    a.push(2.0);
    a.push(-1.0);
    let ba: Vec<_> = a.bitvec.iter().collect();

    // [0, 1, 1]
    b.push(-1.0);
    b.push(1.0);
    b.push(1.0);
    let bb: Vec<_> = b.bitvec.iter().collect();

    assert_eq!(vec![true, true, false], ba);
    assert_eq!(vec![false, true, true], bb);

    let d1 = a.hamming_distance(&b);
    let d2 = b.hamming_distance(&a);

    assert_eq!(d1, d2);
    assert_eq!(2, d1);
}
