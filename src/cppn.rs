use cppn_ext::cppn::{Cppn, CppnNode};
use cppn_ext::position::Position;
use cppn_ext::activation_function::ActivationFunction;
use weight::Weight;
use primal_bit::BitVec;

/// Calculates the behavioral bitvector for `cppn` and all possible inputs `position_pairs`.

fn behavioral_bitvec<P, AF>(cppn: &mut Cppn<CppnNode<AF>, Weight, ()>,
                            position_pairs: &[(P, P)])
                            -> BitVec
    where P: Position,
          AF: ActivationFunction
{
    let outcnt = cppn.output_count();
    assert!(outcnt > 0);
    let mut bitvec = BitVec::from_elem(outcnt * position_pairs.len(), false);
    let mut bitpos = 0;

    for &(ref position1, ref position2) in position_pairs {
        let inputs = [position1.coords(), position2.coords()];
        cppn.process(&inputs[..]);
        for outp in 0..outcnt {
            let output = cppn.read_output(outp).unwrap();
            if output >= 0.0 {
                bitvec.set(bitpos, true);
            }
            bitpos += 1;
        }
    }

    bitvec
}
