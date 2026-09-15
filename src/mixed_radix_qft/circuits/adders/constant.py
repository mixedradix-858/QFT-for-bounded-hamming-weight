"""Constant circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.cuccaro import append_controlled_cuccaro_add


def append_controlled_constant_add(
    circuit: QuantumCircuit,
    control: Qubit,
    value: int,
    constant: Sequence[Qubit],
    target: Sequence[Qubit],
    helper: Qubit,
    mcx_workspace: Sequence[Qubit],
    inverse: bool = False,
) -> None:
    """Controlled compile-time addition modulo ``2**width``."""
    if len(constant) != len(target) or not target:
        raise ValueError("constant and target must have the same positive width")
    reduced = value % (1 << len(target))
    for bit, qubit in enumerate(constant):
        if reduced >> bit & 1:
            circuit.x(qubit)
    append_controlled_cuccaro_add(
        circuit, control, constant, target, helper, mcx_workspace, inverse=inverse
    )
    for bit, qubit in enumerate(constant):
        if reduced >> bit & 1:
            circuit.x(qubit)
