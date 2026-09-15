"""Qfp circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister

from mixed_radix_qft.circuits.local_qft_components.amplification import (
    build_exact_amplified_estimator,
)
from mixed_radix_qft.classical.number_theory import ceil_log2


@dataclass(frozen=True)
class QFPLayout:
    eigen_indices: tuple[int, ...]
    workspace_indices: tuple[int, ...]
    save_indices: tuple[int, ...]


def build_qfp(
    prime: int, arithmetic: str = "explicit-register-sklansky"
) -> tuple[QuantumCircuit, QFPLayout]:
    """Build |Psi_x>|0_work>|0_save> -> |Psi_x>|0_work>|x>."""
    amplified, registers = build_exact_amplified_estimator(prime, arithmetic=arithmetic)
    width = ceil_log2(prime)
    save = QuantumRegister(width, "save")
    circuit = QuantumCircuit(*amplified.qregs, save, name=f"QFP_{prime}")
    amplified_mapping = list(circuit.qubits[: amplified.num_qubits])
    circuit.compose(amplified, qubits=amplified_mapping, inplace=True)
    for source, target in zip(registers.estimate, save, strict=True):
        circuit.cx(source, target)
    circuit.compose(amplified.inverse(), qubits=amplified_mapping, inplace=True)
    eigen_indices = tuple((circuit.find_bit(qubit).index for qubit in registers.eigen))
    save_indices = tuple((circuit.find_bit(qubit).index for qubit in save))
    excluded = set(eigen_indices) | set(save_indices)
    workspace_indices = tuple(
        (index for index in range(circuit.num_qubits) if index not in excluded)
    )
    return (
        circuit,
        QFPLayout(
            eigen_indices=eigen_indices,
            workspace_indices=workspace_indices,
            save_indices=save_indices,
        ),
    )
