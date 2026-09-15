"""Focused regression tests retained from the article implementation."""

from __future__ import annotations

import random
import unittest
from itertools import product
from math import prod

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.lookup import _append_projected_decoder_xor
from mixed_radix_qft.classical.decoder_tables import projected_decoder_plan
from mixed_radix_qft.classical.projected_lookup import (
    project_key,
    project_lookup_table,
    select_injective_bits,
)
from mixed_radix_qft.config import SparseQFTConfig
from mixed_radix_qft.validation.fourier import independent_label
from tests.helpers import basis_output


class ProjectedLookupTests(unittest.TestCase):
    def test_selector_all_sets_and_no_savings_case(self):
        """Exhaust width-three key subsets, including zero rows and full-width cases."""
        for mask in range(1 << 8):
            keys = tuple((key for key in range(8) if mask >> key & 1))
            positions = select_injective_bits(keys, 3)
            packed = [
                sum(((key >> bit & 1) << i for i, bit in enumerate(positions))) for key in keys
            ]
            self.assertEqual(len(set(packed)), len(keys))
            self.assertEqual(positions, select_injective_bits(reversed(keys), 3))
            self.assertLessEqual(len(positions), 3)
        self.assertEqual(select_injective_bits((0, 1, 2, 4, 8), 4), (0, 1, 2, 3))
        self.assertEqual(select_injective_bits((), 0), ())
        self.assertEqual(select_injective_bits((0,), 0), ())
        plan = project_lookup_table(((0, 0), (3, 1)), 2)
        self.assertEqual(len(plan.positions), 1)
        self.assertNotEqual(plan.entries[0][0], plan.entries[1][0])
        for keys, width in (
            ((1, 1), 2),
            ((-1,), 2),
            ((4,), 2),
            ((True,), 1),
            ((1.5,), 2),
            ((), -1),
        ):
            with self.assertRaises(ValueError):
                select_injective_bits(keys, width)
        for value, positions in ((-1, (0,)), (1, (0, 0)), (1, (-1,)), (True, (0,))):
            with self.assertRaises(ValueError):
                project_key(value, positions)
        with self.assertRaises(ValueError):
            project_lookup_table(((0, -1),), 1)

    def test_decoder_every_basis_xor_and_inverse(self):
        """Exhaust promised decoder bases and nonzero XOR targets for all small w."""
        rng = random.Random(271828)
        for moduli in ((2,), (2, 3), (3, 5), (2, 3, 5)):
            m, n = (prod(moduli), (prod(moduli) - 1).bit_length())
            for w, parallel in product(range(n + 1), (False, True)):
                config = SparseQFTConfig.from_moduli(moduli, w=w)
                plan = projected_decoder_plan(config)
                circuit = QuantumCircuit(config.q_y + n)
                _append_projected_decoder_xor(
                    circuit,
                    circuit.qubits[: config.q_y],
                    circuit.qubits[config.q_y :],
                    plan,
                    parallel=parallel,
                )
                inverse = circuit.inverse()
                for x in range(m):
                    if x.bit_count() > w:
                        continue
                    label = independent_label(x, moduli, True)
                    for z in (0, (1 << n) - 1):
                        incoming = label | z << config.q_y
                        expected = label | (z ^ x) << config.q_y
                        self.assertEqual(basis_output(circuit, incoming), expected)
                        self.assertEqual(basis_output(inverse, expected), incoming)
                for _ in range(2):
                    label = rng.getrandbits(circuit.num_qubits)
                    self.assertEqual(basis_output(inverse, basis_output(circuit, label)), label)
