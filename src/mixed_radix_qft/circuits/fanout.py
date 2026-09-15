"""Fanout circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def fanout(circuit, source, targets):
    """CNOT-copy one computational bit coherently to clean private targets."""
    available = [source]
    remaining = list(targets)
    while remaining:
        batch = remaining[: len(available)]
        remaining = remaining[len(batch) :]
        for control, target in zip(available, batch):
            circuit.cx(control, target)
        available.extend(batch)


def _fanout_to_targets(
    circuit: QuantumCircuit, source: Qubit, targets: Sequence[Qubit]
) -> list[tuple[Qubit, Qubit]]:
    """Coherently distribute a computational-basis bit with a CNOT tree."""
    operations: list[tuple[Qubit, Qubit]] = []
    available = [source]
    cursor = 0
    while cursor < len(targets):
        layer_sources = tuple(available)
        new_sources: list[Qubit] = []
        for fanout_source in layer_sources:
            if cursor >= len(targets):
                break
            target = targets[cursor]
            circuit.cx(fanout_source, target)
            operations.append((fanout_source, target))
            new_sources.append(target)
            cursor += 1
        available.extend(new_sources)
    return operations


def _uncompute_fanout(circuit: QuantumCircuit, operations: Sequence[tuple[Qubit, Qubit]]) -> None:
    """Reverse a recorded CNOT distribution tree to erase its copies."""
    for source, target in reversed(operations):
        circuit.cx(source, target)


def _append_controlled_binary_constant(
    circuit: QuantumCircuit, control: Qubit, target: Sequence[Qubit], value: int
) -> None:
    """Prepare ``target = control * value`` with a balanced CNOT fanout."""
    if value < 0 or value >= 1 << len(target):
        raise ValueError("constant does not fit the target register")
    active_targets = [qubit for bit, qubit in enumerate(target) if value >> bit & 1]
    _fanout_to_targets(circuit, control, active_targets)
