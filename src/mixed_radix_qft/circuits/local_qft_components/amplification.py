"""Amplification circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.local_qft_components.estimator import (
    EstimatorRegisters,
    build_uniformized_estimator,
)
from mixed_radix_qft.circuits.local_qft_components.reflections import (
    append_good_phase_oracle,
    append_zero_ancilla_phase_oracle,
)


def build_exact_amplified_estimator(
    prime: int, arithmetic: str = "explicit-register-sklansky"
) -> tuple[QuantumCircuit, EstimatorRegisters]:
    """Build B=A S0 A† Sgood A, amplifying success from 1/4 to one.

    Preserve the Fourier eigenstate; estimator registers remain live for B†."""
    estimator, registers = build_uniformized_estimator(prime, arithmetic=arithmetic)
    circuit = QuantumCircuit(*estimator.qregs, name=f"B_MZ_{prime}")
    mapping = list(circuit.qubits)
    circuit.compose(estimator, qubits=mapping, inplace=True)
    append_good_phase_oracle(circuit, registers, prime)
    circuit.compose(estimator.inverse(), qubits=mapping, inplace=True)
    append_zero_ancilla_phase_oracle(circuit, registers)
    circuit.compose(estimator, qubits=mapping, inplace=True)
    return (circuit, registers)
