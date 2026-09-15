"""Dispatch circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.library import ModularAdderGate

from mixed_radix_qft.circuits.adders.cuccaro import append_cuccaro_add


def _append_modular_sum(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    label: str,
    helper: Qubit | None = None,
) -> None:
    """Add the preserved source into the target modulo 2**width."""
    if len(source) != len(target):
        raise ValueError("source and target addend widths must match")
    if helper is None:
        adder = ModularAdderGate(len(source), label=label)
        circuit.append(adder, [*source, *target])
    else:
        append_cuccaro_add(circuit, source, target, helper)
