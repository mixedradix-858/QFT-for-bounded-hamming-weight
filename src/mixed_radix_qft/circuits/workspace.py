"""Workspace circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumRegister


def fresh(circuit, width):
    """Allocate ``width`` zero-initialized work wires, returning their list."""
    if width == 0:
        return []
    register = QuantumRegister(width, f"work_{len(circuit.qregs)}")
    circuit.add_register(register)
    return list(register)


def undo(circuit, start, stop):
    """Append the inverse of a previously emitted slice without opaque blocks."""
    history = list(circuit.data[start:stop])
    for instruction in reversed(history):
        circuit.append(instruction.operation.inverse(), instruction.qubits)


def append_block(circuit, block, public_wires):
    """Compose a clean block on public wires and fresh private scratch."""
    wires = list(public_wires) + fresh(circuit, block.num_qubits - len(public_wires))
    circuit.compose(block, qubits=wires, inplace=True)
    return wires
