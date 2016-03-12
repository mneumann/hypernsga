use primal_bit::BitVec;

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

    #[inline]
    pub fn push(&mut self, output: f64) {
        if output >= 0.0 {
            self.bitvec.set(self.bitpos, true);
        }
        self.bitpos += 1;
    }
}

