"""Focused regression tests retained from the article implementation."""

from __future__ import annotations

import random
import unittest
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from mixed_radix_qft.circuits.bounded_reduction import append_bounded_reduction
from mixed_radix_qft.circuits.modular_reduction import build_threshold_reduction_circuit
from mixed_radix_qft.circuits.sparse_conversion import build_architecture_fused_conversion
from mixed_radix_qft.config import SparseQFTConfig
from mixed_radix_qft.validation.fourier import independent_label
from mixed_radix_qft.validation.simulation import simulate_sparse_state
from tests.helpers import basis_output

VARIANTS = (
    ("cuccaro", "local"),
    ("cuccaro", "deferred"),
    ("prefix", "local"),
    ("prefix", "deferred"),
)


def clean_outer_reduction(modulus, w, backend, cleanup):
    """Wrap either R mode in the same compute-copy-uncompute interface."""
    width, output_width = ((w * (modulus - 1)).bit_length(), (modulus - 1).bit_length())
    source, target = (QuantumRegister(width, "s"), QuantumRegister(output_width, "r"))
    circuit = QuantumCircuit(source, target)
    result = append_bounded_reduction(circuit, source, modulus, w, backend=backend, cleanup=cleanup)
    history = list(circuit.data)
    for left, right in zip(result, target):
        circuit.cx(left, right)
    for instruction in reversed(history):
        circuit.append(instruction.operation.inverse(), instruction.qubits)
    return circuit


class BoundedReductionTests(unittest.TestCase):
    def test_clean_reduction_all_sums_and_physical_inverse(self):
        """Exhaust valid sums, boundaries, mod 2 and physical inverse samples."""
        rng = random.Random(271828)
        for p, w, backend in product((2, 3, 5, 7, 11), (1, 2, 3, 4), ("cuccaro", "prefix")):
            circuit, result = build_threshold_reduction_circuit(
                p, w, True, reduction_backend=backend
            )
            inverse = circuit.inverse()
            for s in range(w * (p - 1) + 1):
                expected = s | sum(((s % p >> bit & 1) << q for bit, q in enumerate(result)))
                with self.subTest(p=p, w=w, backend=backend, s=s):
                    self.assertEqual(basis_output(circuit, s), expected)
                    self.assertEqual(basis_output(inverse, expected), s)
            for _ in range(3):
                incoming = rng.getrandbits(circuit.num_qubits)
                self.assertEqual(basis_output(inverse, basis_output(circuit, incoming)), incoming)

    def test_deferred_cleanup_and_complex_arithmetic(self):
        """Check every variant with XOR targets and coherent outer cleanup."""
        for p, w in ((2, 1), (3, 2), (7, 2), (5, 4)):
            width = (w * (p - 1)).bit_length()
            for backend, cleanup in VARIANTS:
                circuit = clean_outer_reduction(p, w, backend, cleanup)
                for s, target in product(
                    range(w * (p - 1) + 1), (0, (1 << (p - 1).bit_length()) - 1)
                ):
                    incoming = s | target << width
                    self.assertEqual(basis_output(circuit, incoming), s | (target ^ s % p) << width)
                vector = np.array([complex(s + 1, 2 - s) for s in range(w * (p - 1) + 1)])
                vector /= np.linalg.norm(vector)
                actual = simulate_sparse_state(
                    circuit, dict(enumerate(vector)), max_states=64, prune_tolerance=0
                )
                expected = {s | s % p << width: a for s, a in enumerate(vector)}
                self.assertEqual(actual, expected)

    def test_conversion_bases_and_backend_combinations(self):
        """Check eta, zero input/work, and inverse for all small promised bases."""
        for moduli in ((2, 3), (3, 5)):
            for w in (0, 1, 2, (np.prod(moduli).item() - 1).bit_length()):
                config = SparseQFTConfig.from_moduli(moduli, w=w)
                for backend, cleanup in VARIANTS:
                    for weighted, decoder in (
                        ("wallace-qfa2", "lookup"),
                        ("cuccaro", "crt-kogge-stone"),
                    ):
                        circuit = build_architecture_fused_conversion(
                            config,
                            explicit_primitives=True,
                            decoder=decoder,
                            weighted_sum_backend=weighted,
                            reduction_backend=backend,
                            reduction_cleanup=cleanup,
                        )
                        inverse = circuit.inverse()
                        for x in range(config.crt_modulus):
                            if x.bit_count() <= w:
                                expected = independent_label(x, moduli, True) << config.n
                                self.assertEqual(basis_output(circuit, x), expected)
                                self.assertEqual(basis_output(inverse, expected), x)
