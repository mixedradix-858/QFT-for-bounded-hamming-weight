"""Kogge stone circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.prefix_support import (
    _record_self_inverse_gate,
    _uncompute_recorded_gates,
)


def kogge_stone_prefix_workspace_size(width: int) -> int:
    """Clean workspace for a reversible bounded-fanout prefix network."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    nodes = 0
    stride = 1
    while stride < width:
        nodes += width - stride
        stride *= 2
    return 2 * width + 2 * nodes


def _append_kogge_stone_prefix_compute(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    workspace: Sequence[Qubit],
    *,
    complement_right: bool,
) -> tuple[
    tuple[Qubit, ...],
    tuple[Qubit, ...],
    tuple[Qubit, ...],
    tuple[tuple[str, tuple[Qubit, ...]], ...],
]:
    """Compute all prefixes with constant fan-out per prefix level."""
    left = tuple(left)
    right = tuple(right)
    width = len(left)
    if len(right) != width or width <= 0:
        raise ValueError("operands must have the same positive width")
    required = kogge_stone_prefix_workspace_size(width)
    if len(workspace) < required:
        raise ValueError(f"a {width}-bit Kogge-Stone prefix needs {required} clean qubits")
    cursor = 0
    leaf_p = tuple(workspace[cursor : cursor + width])
    cursor += width
    leaf_g = tuple(workspace[cursor : cursor + width])
    cursor += width
    operations: list[tuple[str, tuple[Qubit, ...]]] = []
    if complement_right:
        for propagate in leaf_p:
            _record_self_inverse_gate(circuit, operations, "x", propagate)
    for source, propagate in zip(left, leaf_p, strict=True):
        _record_self_inverse_gate(circuit, operations, "cx", source, propagate)
    for source, propagate in zip(right, leaf_p, strict=True):
        _record_self_inverse_gate(circuit, operations, "cx", source, propagate)
    if complement_right:
        for qubit in right:
            _record_self_inverse_gate(circuit, operations, "x", qubit)
    for left_qubit, right_qubit, generate in zip(left, right, leaf_g, strict=True):
        _record_self_inverse_gate(circuit, operations, "ccx", left_qubit, right_qubit, generate)
    if complement_right:
        for qubit in right:
            _record_self_inverse_gate(circuit, operations, "x", qubit)
    current_p = list(leaf_p)
    current_g = list(leaf_g)
    stride = 1
    while stride < width:
        level_outputs: dict[int, tuple[Qubit, Qubit]] = {}
        for target_index in range(stride, width):
            propagate = workspace[cursor]
            generate = workspace[cursor + 1]
            cursor += 2
            level_outputs[target_index] = (propagate, generate)
        for target_index in range(stride, width):
            generate = level_outputs[target_index][1]
            _record_self_inverse_gate(circuit, operations, "cx", current_g[target_index], generate)
        for target_index in range(stride, width):
            generate = level_outputs[target_index][1]
            _record_self_inverse_gate(
                circuit,
                operations,
                "ccx",
                current_p[target_index],
                current_g[target_index - stride],
                generate,
            )
        for colour in (0, 1):
            for target_index in range(stride, width):
                if target_index // stride % 2 != colour:
                    continue
                propagate = level_outputs[target_index][0]
                _record_self_inverse_gate(
                    circuit,
                    operations,
                    "ccx",
                    current_p[target_index],
                    current_p[target_index - stride],
                    propagate,
                )
        for target_index, (propagate, generate) in level_outputs.items():
            current_p[target_index] = propagate
            current_g[target_index] = generate
        stride *= 2
    if cursor != required:
        raise AssertionError(f"Kogge-Stone workspace accounting used {cursor}, expected {required}")
    return (leaf_p, tuple(current_p), tuple(current_g), tuple(operations))


def _append_kogge_stone_sum_xor(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    output: Sequence[Qubit],
    prefix_workspace: Sequence[Qubit],
    *,
    complement_right: bool,
    carry_in: int,
) -> None:
    """XOR a bounded-fanout prefix sum into ``output`` and clean work."""
    left = tuple(left)
    right = tuple(right)
    output = tuple(output)
    width = len(left)
    if len(right) != width or len(output) != width or width <= 0:
        raise ValueError("sum registers must have the same positive width")
    if carry_in not in {0, 1}:
        raise ValueError("carry_in must be the classical bit 0 or 1")
    leaf_p, prefixes_p, prefixes_g, operations = _append_kogge_stone_prefix_compute(
        circuit, left, right, prefix_workspace, complement_right=complement_right
    )
    for propagate, target in zip(leaf_p, output, strict=True):
        circuit.cx(propagate, target)
    if carry_in:
        circuit.x(output[0])
    for bit in range(1, width):
        circuit.cx(prefixes_g[bit - 1], output[bit])
    if carry_in:
        for bit in range(1, width):
            circuit.cx(prefixes_p[bit - 1], output[bit])
    _uncompute_recorded_gates(circuit, operations)


def append_kogge_stone_sum(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    output: Sequence[Qubit],
    workspace: Sequence[Qubit],
) -> None:
    """Append an ``O(log n)`` bounded-fanout out-of-place sum."""
    if len(left) != len(right) or len(left) != len(output) or (not left):
        raise ValueError("left, right, and output need equal positive width")
    required = kogge_stone_prefix_workspace_size(len(left))
    if len(workspace) < required:
        raise ValueError("insufficient Kogge-Stone workspace")
    _append_kogge_stone_sum_xor(
        circuit, left, right, output, workspace[:required], complement_right=False, carry_in=0
    )


def append_kogge_stone_difference(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    output: Sequence[Qubit],
    workspace: Sequence[Qubit],
) -> None:
    """Append an ``O(log n)`` bounded-fanout out-of-place difference."""
    if len(left) != len(right) or len(left) != len(output) or (not left):
        raise ValueError("left, right, and output need equal positive width")
    required = kogge_stone_prefix_workspace_size(len(left))
    if len(workspace) < required:
        raise ValueError("insufficient Kogge-Stone workspace")
    _append_kogge_stone_sum_xor(
        circuit, left, right, output, workspace[:required], complement_right=True, carry_in=1
    )
