"""Comparators circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def comparator_workspace_size(width: int) -> int:
    """Return 4*width-2 clean bits for a balanced greater/equal comparison tree."""
    if width <= 0:
        raise ValueError("comparator width must be positive")
    return 4 * width - 2


def append_log_depth_constant_comparator(
    circuit: QuantumCircuit,
    state: Sequence[Qubit],
    flag: Qubit,
    value: int,
    workspace: Sequence[Qubit],
) -> None:
    """Append flag ^= [state >= value] with a balanced comparison tree."""
    state = tuple(state)
    width = len(state)
    if width <= 0:
        raise ValueError("state width must be positive")
    if value < 0 or value >= 1 << width:
        raise ValueError("comparison value does not fit the state register")
    required = comparator_workspace_size(width)
    if len(workspace) < required:
        raise ValueError(f"a {width}-bit comparator requires {required} clean work qubits")
    leaves: list[tuple[Qubit, Qubit]] = []
    leaf_operations: list[tuple[int, Qubit, Qubit, Qubit]] = []
    cursor = 0
    for bit_index in reversed(range(width)):
        greater = workspace[cursor]
        equal = workspace[cursor + 1]
        cursor += 2
        state_qubit = state[bit_index]
        constant_bit = value >> bit_index & 1
        if constant_bit == 0:
            circuit.x(equal)
            circuit.cx(state_qubit, greater)
            circuit.cx(state_qubit, equal)
        else:
            circuit.cx(state_qubit, equal)
        leaf_operations.append((constant_bit, state_qubit, greater, equal))
        leaves.append((greater, equal))
    current = leaves
    internal_operations: list[tuple[Qubit, Qubit, Qubit, Qubit, Qubit, Qubit]] = []
    while len(current) > 1:
        next_level: list[tuple[Qubit, Qubit]] = []
        for node_index in range(0, len(current), 2):
            if node_index + 1 == len(current):
                next_level.append(current[node_index])
                continue
            greater_high, equal_high = current[node_index]
            greater_low, equal_low = current[node_index + 1]
            greater_out = workspace[cursor]
            equal_out = workspace[cursor + 1]
            cursor += 2
            circuit.cx(greater_high, greater_out)
            circuit.ccx(equal_high, greater_low, greater_out)
            circuit.ccx(equal_high, equal_low, equal_out)
            internal_operations.append(
                (greater_high, equal_high, greater_low, equal_low, greater_out, equal_out)
            )
            next_level.append((greater_out, equal_out))
        current = next_level
    greater_root, equal_root = current[0]
    circuit.cx(greater_root, flag)
    circuit.cx(equal_root, flag)
    for greater_high, equal_high, greater_low, equal_low, greater_out, equal_out in reversed(
        internal_operations
    ):
        circuit.ccx(equal_high, equal_low, equal_out)
        circuit.ccx(equal_high, greater_low, greater_out)
        circuit.cx(greater_high, greater_out)
    for constant_bit, state_qubit, greater, equal in reversed(leaf_operations):
        if constant_bit == 0:
            circuit.cx(state_qubit, equal)
            circuit.cx(state_qubit, greater)
            circuit.x(equal)
        else:
            circuit.cx(state_qubit, equal)
