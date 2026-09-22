"""Carry-save CRT cleanup: arithmetic, physical cleanup and Fourier interference."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from itertools import product

import numpy as np
from qiskit import QuantumCircuit

from mixed_radix_qft.algorithms.sparse import build_sparse_qft
from mixed_radix_qft.circuits.decoders import (
    _append_crt_inverse_decoder_xor,
    build_crt_inverse_decoder_block,
)
from mixed_radix_qft.circuits.sparse_conversion import build_architecture_fused_conversion
from mixed_radix_qft.cli import main
from mixed_radix_qft.config import SparseQFTConfig
from mixed_radix_qft.validation.fourier import independent_dft, independent_label, sparse_error
from mixed_radix_qft.validation.simulation import simulate_sparse_state
from tests.helpers import basis_output


def decoder_circuit(config, backend):
    circuit = QuantumCircuit(config.q_y + config.n)
    _append_crt_inverse_decoder_xor(
        circuit,
        circuit.qubits[: config.q_y],
        circuit.qubits[config.q_y :],
        config,
        prefix="kogge-stone",
        sum_backend=backend,
    )
    return circuit


class CarrySaveDecoderTests(unittest.TestCase):
    def test_valid_tuples_arbitrary_targets_and_clean_work(self):
        # Exhaust every valid CRT tuple, including non-sparse x; do not use
        # production CRT inversion as the oracle. Equality checks all work bits.
        for factors in ((2,), (3,), (2, 3), (3, 5), (2, 3, 5), (2, 3, 5, 7)):
            config = SparseQFTConfig.from_moduli(factors, w=1)
            circuit = decoder_circuit(config, "wallace-qfa2")
            targets = {0, (1 << config.n) - 1, sum(1 << b for b in range(0, config.n, 2))}
            for fields in product(*(range(p) for p in factors)):
                packed, offset = 0, 0
                for value, width in zip(fields, config.widths):
                    packed |= value << offset
                    offset += width
                expected = sum(config.crt_modulus // p * y for p, y in zip(factors, fields))
                expected %= config.crt_modulus
                for target in targets:
                    initial = packed | (target << config.q_y)
                    self.assertEqual(
                        basis_output(circuit, initial),
                        packed | ((target ^ expected) << config.q_y),
                    )

    def test_physical_extension_matches_old_decoder_and_is_involutive(self):
        # Unused local encodings have no CRT specification; preserve the old
        # backend's extension and check reversibility, including nonzero X.
        for factors in ((3,), (2, 3), (3, 5), (2, 3, 5)):
            config = SparseQFTConfig.from_moduli(factors, w=1)
            old = decoder_circuit(config, "prefix-tree")
            new = decoder_circuit(config, "wallace-qfa2")
            for packed in range(1 << config.q_y):
                initial = packed | (((1 << config.n) - 1) << config.q_y)
                actual = basis_output(new, initial)
                self.assertEqual(actual, basis_output(old, initial))
                self.assertEqual(basis_output(new, actual), initial)

    def test_fused_conversion_complex_superposition_and_inverse(self):
        config = SparseQFTConfig.from_moduli((2, 3, 5), w=2)
        circuit = build_architecture_fused_conversion(
            config,
            decoder="crt-wallace-kogge-stone",
            explicit_primitives=True,
            reduction_backend="prefix",
            reduction_cleanup="deferred",
        )
        vector = {x: complex(x + 1, 2 - x) for x in range(30) if x.bit_count() <= 2}
        norm = np.sqrt(sum(abs(a) ** 2 for a in vector.values()))
        vector = {x: a / norm for x, a in vector.items()}
        expected = {
            independent_label(x, config.moduli, True) << config.n: a for x, a in vector.items()
        }
        actual = simulate_sparse_state(circuit, vector, prune_tolerance=0)
        self.assertLess(sparse_error(actual, expected), 1e-12)
        self.assertLess(
            sparse_error(
                simulate_sparse_state(circuit.inverse(), actual, prune_tolerance=0), vector
            ),
            1e-12,
        )

    def test_terminal_and_coherent_qft_complex_states(self):
        for w, coherent in product((0, 1, 2, 3), (False, True)):
            config = SparseQFTConfig.from_moduli((2, 3), w=w)
            circuit = build_sparse_qft(
                config,
                decoder="crt-wallace-kogge-stone",
                coherent=coherent,
                reduction_backend="prefix",
                reduction_cleanup="deferred",
            )
            vector = np.array(
                [complex(x + 1, 2 - x) if x.bit_count() <= w else 0 for x in range(6)]
            )
            vector /= np.linalg.norm(vector)
            expected = {
                y if coherent else independent_label(y, config.moduli, False) << config.n: a
                for y, a in enumerate(independent_dft(6) @ vector)
            }
            actual = simulate_sparse_state(circuit, dict(enumerate(vector)), prune_tolerance=1e-13)
            self.assertLess(sparse_error(actual, expected), 1e-8)

    def test_cli_selects_and_simulates_new_decoder(self):
        output = StringIO()
        with redirect_stdout(output):
            main(
                [
                    "sparse",
                    "--moduli",
                    "2",
                    "3",
                    "--w",
                    "2",
                    "--decoder",
                    "crt-wallace-kogge-stone",
                    "--simulate",
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(report["configuration"]["decoder"], "crt-wallace-kogge-stone")
        self.assertTrue(report["validation"]["passed"])

    def test_invalid_backend_combinations_rejected(self):
        config = SparseQFTConfig.from_moduli((2, 3), w=1)
        for prefix, backend in (("sklansky", "wallace-qfa2"), ("kogge-stone", "bad")):
            with self.assertRaises(ValueError):
                build_crt_inverse_decoder_block(config, prefix, sum_backend=backend)
