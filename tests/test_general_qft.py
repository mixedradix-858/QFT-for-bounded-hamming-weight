"""Focused regression tests retained from the article implementation."""

from __future__ import annotations

import unittest
from itertools import product

import numpy as np

from mixed_radix_qft.algorithms.general import build_general_mixed_qft
from mixed_radix_qft.algorithms.sparse import build_sparse_qft as build_sparse_coherent_qft
from mixed_radix_qft.circuits.a import build_general_a
from mixed_radix_qft.circuits.c import build_general_c
from mixed_radix_qft.config import SparseQFTConfig
from mixed_radix_qft.registers import crt_layout
from mixed_radix_qft.validation.fourier import independent_dft, independent_label, sparse_error
from mixed_radix_qft.validation.simulation import simulate_sparse_state
from tests.helpers import basis_output, error_gram


class GeneralTransformTests(unittest.TestCase):
    def test_general_c_a_all_labels_and_physical_inverse(self):
        """Check general C/A, injectivity, ordinary CRT inverse and invalid inputs."""
        for moduli in ((2,), (3, 2), (3, 5), (4, 3), (2, 3, 5)):
            m, n, widths, _ = crt_layout(moduli)
            conversion, multiplier = (build_general_c(moduli), build_general_a(moduli))
            for x in range(m):
                chi = independent_label(x, moduli, False)
                eta = independent_label(x, moduli, True)
                self.assertEqual(basis_output(conversion, x), chi << n)
                self.assertEqual(basis_output(conversion.inverse(), chi << n), x)
                self.assertEqual(basis_output(multiplier, chi), eta)
                self.assertEqual(basis_output(multiplier.inverse(), eta), chi)
            for invalid in range(m, 1 << n):
                self.assertEqual(
                    basis_output(conversion.inverse(), basis_output(conversion, invalid)), invalid
                )
        for factors in ((), (2, 4), (1, 3), (3, 3)):
            with self.assertRaises(ValueError):
                build_general_c(factors)

    def test_general_coherent_and_terminal_complex_states(self):
        """Compare explicit MZ mixed-radix to an independent DFT, with cleanup."""
        m, factors = (6, (2, 3))
        rng = np.random.default_rng(271828)
        amplitudes = rng.normal(size=m) + 1j * rng.normal(size=m)
        amplitudes /= np.linalg.norm(amplitudes)
        expected_vector = independent_dft(m) @ amplitudes
        for coherent in (False, True):
            circuit = build_general_mixed_qft(factors, coherent=coherent)
            diagnostics = {}
            actual = simulate_sparse_state(
                circuit, dict(enumerate(amplitudes)), prune_tolerance=1e-13, diagnostics=diagnostics
            )
            self.assertLess(diagnostics.get("discarded_l2_bound", 0), 1e-08)
            expected = {
                y if coherent else independent_label(y, factors, False) << 3: a
                for y, a in enumerate(expected_vector)
            }
            self.assertLess(sparse_error(actual, expected), 1e-08)
            if coherent:
                columns = [
                    simulate_sparse_state(circuit, {x: 1}, prune_tolerance=1e-13) for x in range(m)
                ]
                self.assertLess(error_gram(columns, m), 1e-08)
            else:
                distribution = np.zeros(m)
                inverse = {independent_label(y, factors, False): y for y in range(m)}
                for label, amplitude in actual.items():
                    packed = label >> 3 & 7
                    self.assertIn(packed, inverse)
                    distribution[inverse[packed]] += abs(amplitude) ** 2
                np.testing.assert_allclose(
                    distribution, abs(expected_vector) ** 2, atol=1e-08, rtol=0
                )
            self.assertFalse(
                set(circuit.count_ops())
                - {"x", "cx", "ccx", "h", "p", "cp", "ry", "cry", "swap", "z"}
            )

    def test_sparse_coherent_all_arithmetic_variants(self):
        """Check sparse input promises with general post-QFT decoding."""
        for w, decoder, backend in product(
            (0, 1, 2),
            ("lookup", "crt-inverse", "crt-kogge-stone", "crt-wallace-kogge-stone"),
            ("wallace-qfa2", "cuccaro"),
        ):
            config = SparseQFTConfig.from_moduli((2, 3), w=w)
            circuit = build_sparse_coherent_qft(
                config, weighted_sum_backend=backend, decoder=decoder
            )
            vector = np.array(
                [complex(x + 1, 1 - x) if x.bit_count() <= w else 0 for x in range(6)]
            )
            vector /= np.linalg.norm(vector)
            actual = simulate_sparse_state(
                circuit, {x: a for x, a in enumerate(vector) if a}, prune_tolerance=1e-13
            )
            expected = dict(enumerate(independent_dft(6) @ vector))
            self.assertLess(sparse_error(actual, expected), 1e-08)
