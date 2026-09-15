"""Focused regression tests retained from the article implementation."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mixed_radix_qft.validation.reference import require_dense_memory
from mixed_radix_qft.validation.simulation import (
    simulate_classical_reversible_circuit,
    simulate_sparse_state,
)


class SimulatorRegressionTests(unittest.TestCase):
    def test_sparse_simulator_against_qiskit(self) -> None:
        """Verify every custom gate, its inverse and global phase against Qiskit."""
        circuit = QuantumCircuit(6, global_phase=0.37)
        circuit.x(4)
        circuit.h(1)
        circuit.cx(1, 3)
        circuit.ccx(3, 4, 0)
        circuit.swap(0, 5)
        circuit.p(-0.23, 2)
        circuit.cp(0.71, 5, 1)
        circuit.ry(-0.49, 0)
        circuit.cry(1.37, 4, 2)
        circuit.z(3)
        rng = np.random.default_rng(1701)
        vector = rng.normal(size=64) + 1j * rng.normal(size=64)
        vector /= np.linalg.norm(vector)
        initial = dict(enumerate(vector))
        require_dense_memory(6)
        for candidate in (circuit, circuit.inverse(), circuit.compose(circuit.inverse())):
            actual = simulate_sparse_state(candidate, initial, prune_tolerance=0)
            expected = Statevector(vector).evolve(candidate).data
            self.assertLess(
                np.linalg.norm(np.array([actual.get(i, 0) for i in range(64)]) - expected), 1e-12
            )

    def test_classical_simulator_truth_tables_and_invalid_input(self) -> None:
        """Verify the bit simulator on every physical basis of a small circuit."""
        circuit = QuantumCircuit(4)
        circuit.x(3)
        circuit.cx(3, 1)
        circuit.ccx(1, 0, 2)
        require_dense_memory(4)
        for index in range(16):
            result = simulate_classical_reversible_circuit(
                circuit, [index >> b & 1 for b in range(4)]
            )
            expected = np.zeros(16, complex)
            expected[sum((bit << b for b, bit in enumerate(result)))] = 1
            np.testing.assert_allclose(
                Statevector.from_int(index, 16).evolve(circuit).data, expected, rtol=0, atol=1e-12
            )
        for bits in ([2], [0] * 5):
            with self.assertRaises(ValueError):
                simulate_classical_reversible_circuit(circuit, bits)
        circuit.global_phase = 0.2
        with self.assertRaises(ValueError):
            simulate_classical_reversible_circuit(circuit, [])
