"""Prefix support circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def _record_self_inverse_gate(
    circuit: QuantumCircuit,
    operations: list[tuple[str, tuple[Qubit, ...]]],
    name: str,
    *qubits: Qubit,
) -> None:
    """Append an X, CX or CCX and record it for exact reversal."""
    if name == "x":
        circuit.x(qubits[0])
    elif name == "cx":
        circuit.cx(qubits[0], qubits[1])
    elif name == "ccx":
        circuit.ccx(qubits[0], qubits[1], qubits[2])
    else:
        raise AssertionError(f"unsupported recorded gate {name!r}")
    operations.append((name, tuple(qubits)))


def _uncompute_recorded_gates(
    circuit: QuantumCircuit, operations: Sequence[tuple[str, tuple[Qubit, ...]]]
) -> None:
    """Reverse a history of self-inverse gates to erase intermediate work."""
    for name, qubits in reversed(operations):
        if name == "x":
            circuit.x(qubits[0])
        elif name == "cx":
            circuit.cx(qubits[0], qubits[1])
        elif name == "ccx":
            circuit.ccx(qubits[0], qubits[1], qubits[2])
        else:
            raise AssertionError(f"unsupported recorded gate {name!r}")


def _append_balanced_fanout(
    circuit: QuantumCircuit,
    source: Qubit,
    count: int,
    workspace: Sequence[Qubit],
    operations: list[tuple[str, tuple[Qubit, ...]]],
) -> tuple[tuple[Qubit, ...], tuple[tuple[Qubit, Qubit], ...]]:
    """Create ``count`` usable controls with a logarithmic-depth CNOT tree."""
    if count <= 0:
        return ((), ())
    if len(workspace) < count - 1:
        raise ValueError("insufficient Sklansky fanout workspace")
    controls: list[Qubit] = [source]
    history: list[tuple[Qubit, Qubit]] = []
    cursor = 0
    while len(controls) < count:
        existing = tuple(controls)
        additions = min(len(existing), count - len(controls))
        for control in existing[:additions]:
            target = workspace[cursor]
            cursor += 1
            _record_self_inverse_gate(circuit, operations, "cx", control, target)
            controls.append(target)
            history.append((control, target))
    return (tuple(controls), tuple(history))


def _append_cx_swap(circuit: QuantumCircuit, left: Sequence[Qubit], right: Sequence[Qubit]) -> None:
    """Exchange equal-width words using three parallel CNOT layers."""
    for left_qubit, right_qubit in zip(left, right, strict=True):
        circuit.cx(left_qubit, right_qubit)
    for left_qubit, right_qubit in zip(left, right, strict=True):
        circuit.cx(right_qubit, left_qubit)
    for left_qubit, right_qubit in zip(left, right, strict=True):
        circuit.cx(left_qubit, right_qubit)
