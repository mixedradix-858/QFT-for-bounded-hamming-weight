"""Counter roots retain promised counts and reverse their entire history."""

import unittest

from mixed_radix_qft.circuits.counters import build_counter_forest_circuit
from tests.helpers import basis_output


class CounterTests(unittest.TestCase):
    def test_explicit_counter_roots_and_inverse(self):
        for n, modulus, w in ((4, 3, 0), (4, 3, 2), (5, 5, 3)):
            circuit, roots = build_counter_forest_circuit(n, modulus, w)
            self.assertFalse(set(circuit.count_ops()) - {"x", "cx", "ccx"})
            for value in range(1 << n):
                if value.bit_count() > w:
                    continue
                actual = basis_output(circuit, value)
                self.assertEqual(actual & ((1 << n) - 1), value)
                for residue, wires in roots:
                    expected = sum(
                        (value >> i) & 1 for i in range(n) if pow(2, i, modulus) == residue
                    )
                    count = sum(((actual >> wire) & 1) << i for i, wire in enumerate(wires))
                    self.assertEqual(count, expected)
                self.assertEqual(basis_output(circuit.inverse(), actual), value)
