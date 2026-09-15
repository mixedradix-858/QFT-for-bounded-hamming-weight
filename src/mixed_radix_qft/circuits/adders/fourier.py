"""Fourier circuits and mathematical helpers."""

from __future__ import annotations

from math import pi
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import QFTGate


def append_fourier_controlled_constant_additions(
    circuit: QuantumCircuit,
    controls: Sequence[Qubit],
    values: Sequence[int],
    target: Sequence[Qubit],
) -> None:
    """Add several controlled constants with one shared Fourier transform."""
    controls = tuple(controls)
    values = tuple(values)
    target = tuple(target)
    if len(controls) != len(values):
        raise ValueError("controls and values must have the same length")
    if not target:
        raise ValueError("target must be non-empty")
    if not controls:
        return
    width = len(target)
    order = 1 << width
    circuit.compose(QFTGate(width).definition, qubits=target, inplace=True)
    for control, value in zip(controls, values, strict=True):
        reduced = value % order
        for bit, target_qubit in enumerate(target):
            phase_numerator = (reduced << bit) % order
            if phase_numerator:
                circuit.cp(2 * pi * phase_numerator / order, control, target_qubit)
    circuit.compose(QFTGate(width).inverse().definition, qubits=target, inplace=True)


def build_fourier_controlled_constant_additions(
    width: int, values: Sequence[int]
) -> QuantumCircuit:
    """Map |controls>|y> to |controls>|y+sum_i c_i*controls_i mod 2**b>.

    Controls are preserved; shared QFTs require no ancillary register.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    values = tuple(values)
    if not values:
        raise ValueError("at least one controlled value is required")
    controls = QuantumRegister(len(values), "controls")
    target = QuantumRegister(width, "target")
    circuit = QuantumCircuit(controls, target, name=f"fourier_controlled_constants_{width}")
    append_fourier_controlled_constant_additions(circuit, controls, values, target)
    return circuit
