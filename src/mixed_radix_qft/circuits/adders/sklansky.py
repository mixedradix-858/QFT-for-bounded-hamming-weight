"""Sklansky circuits and mathematical helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.prefix_support import (
    _append_balanced_fanout,
    _append_cx_swap,
    _record_self_inverse_gate,
    _uncompute_recorded_gates,
)


def _sklansky_groups(width: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return all ``(lower-prefix boundary, upper targets)`` tree groups."""
    groups: list[tuple[int, tuple[int, ...]]] = []
    half = 1
    while half < width:
        block = 2 * half
        for start in range(0, width, block):
            boundary = start + half - 1
            targets = tuple(range(start + half, min(start + block, width)))
            if targets:
                groups.append((boundary, targets))
        half *= 2
    return tuple(groups)


def sklansky_prefix_workspace_size(width: int) -> int:
    """Clean ancillas used by the reversible Sklansky prefix network."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    nodes = 0
    copies = 0
    for _, targets in _sklansky_groups(width):
        nodes += len(targets)
        copies += len(targets) - 1
    return 2 * width + 2 * nodes + 2 * copies


def sklansky_inplace_workspace_size(width: int) -> int:
    """Workspace for a clean in-place Sklansky add/subtract."""
    return width + sklansky_prefix_workspace_size(width)


def _append_sklansky_prefix_compute(
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
    """Compute all Sklansky prefixes and return a reversible gate history."""
    left = tuple(left)
    right = tuple(right)
    width = len(left)
    if len(right) != width or width <= 0:
        raise ValueError("operands must have the same positive width")
    required = sklansky_prefix_workspace_size(width)
    if len(workspace) < required:
        raise ValueError(f"a {width}-bit Sklansky prefix needs {required} clean qubits")
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
    half = 1
    while half < width:
        block = 2 * half
        level_groups: list[
            tuple[
                tuple[int, ...],
                tuple[Qubit, ...],
                tuple[Qubit, ...],
                tuple[tuple[Qubit, Qubit], ...],
                tuple[tuple[Qubit, Qubit], ...],
            ]
        ] = []
        level_outputs: dict[int, tuple[Qubit, Qubit]] = {}
        for start in range(0, width, block):
            boundary = start + half - 1
            targets = tuple(range(start + half, min(start + block, width)))
            if not targets:
                continue
            copy_count = len(targets) - 1
            p_copy_space = workspace[cursor : cursor + copy_count]
            cursor += copy_count
            g_copy_space = workspace[cursor : cursor + copy_count]
            cursor += copy_count
            p_controls, p_history = _append_balanced_fanout(
                circuit, current_p[boundary], len(targets), p_copy_space, operations
            )
            g_controls, g_history = _append_balanced_fanout(
                circuit, current_g[boundary], len(targets), g_copy_space, operations
            )
            level_groups.append((targets, p_controls, g_controls, p_history, g_history))
        for targets, _, _, _, _ in level_groups:
            for target_index in targets:
                propagate = workspace[cursor]
                generate = workspace[cursor + 1]
                cursor += 2
                level_outputs[target_index] = (propagate, generate)
        for targets, _, _, _, _ in level_groups:
            for target_index in targets:
                generate = level_outputs[target_index][1]
                _record_self_inverse_gate(
                    circuit, operations, "cx", current_g[target_index], generate
                )
        for targets, p_controls, _, _, _ in level_groups:
            for target_index, lower_p in zip(targets, p_controls, strict=True):
                propagate = level_outputs[target_index][0]
                _record_self_inverse_gate(
                    circuit, operations, "ccx", current_p[target_index], lower_p, propagate
                )
        for targets, _, g_controls, _, _ in level_groups:
            for target_index, lower_g in zip(targets, g_controls, strict=True):
                generate = level_outputs[target_index][1]
                _record_self_inverse_gate(
                    circuit, operations, "ccx", current_p[target_index], lower_g, generate
                )
        for _, _, _, p_history, g_history in reversed(level_groups):
            for control, target in reversed(g_history):
                _record_self_inverse_gate(circuit, operations, "cx", control, target)
            for control, target in reversed(p_history):
                _record_self_inverse_gate(circuit, operations, "cx", control, target)
        for target_index, (propagate, generate) in level_outputs.items():
            current_p[target_index] = propagate
            current_g[target_index] = generate
        half *= 2
    if cursor != required:
        raise AssertionError(f"Sklansky workspace accounting used {cursor}, expected {required}")
    return (leaf_p, tuple(current_p), tuple(current_g), tuple(operations))


def _append_sklansky_sum_xor(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    output: Sequence[Qubit],
    prefix_workspace: Sequence[Qubit],
    *,
    complement_right: bool,
    carry_in: int,
    carry_out: Qubit | None = None,
) -> None:
    """XOR an out-of-place sum into ``output`` and clean every ancilla."""
    left = tuple(left)
    right = tuple(right)
    output = tuple(output)
    width = len(left)
    if len(right) != width or len(output) != width or width <= 0:
        raise ValueError("sum registers must have the same positive width")
    if carry_in not in {0, 1}:
        raise ValueError("carry_in must be the classical bit 0 or 1")
    leaf_p, prefixes_p, prefixes_g, operations = _append_sklansky_prefix_compute(
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
    if carry_out is not None:
        circuit.cx(prefixes_g[-1], carry_out)
        if carry_in:
            circuit.cx(prefixes_p[-1], carry_out)
    _uncompute_recorded_gates(circuit, operations)


def _append_clean_sklansky_add(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    carry_out: Qubit | None,
    workspace: Sequence[Qubit],
) -> None:
    """Implement an in-place add using two clean out-of-place prefixes."""
    source = tuple(source)
    target = tuple(target)
    width = len(source)
    required = sklansky_inplace_workspace_size(width)
    if len(target) != width or width <= 0:
        raise ValueError("source and target must have equal positive width")
    if len(workspace) < required:
        raise ValueError(f"a {width}-bit in-place Sklansky add needs {required} qubits")
    scratch = tuple(workspace[:width])
    prefix = tuple(workspace[width:required])
    _append_sklansky_sum_xor(
        circuit,
        source,
        target,
        scratch,
        prefix,
        complement_right=False,
        carry_in=0,
        carry_out=carry_out,
    )
    _append_cx_swap(circuit, target, scratch)
    _append_sklansky_sum_xor(
        circuit, target, source, scratch, prefix, complement_right=True, carry_in=1
    )


@lru_cache(maxsize=None)
def sklansky_fixed_adder(width: int) -> QuantumCircuit:
    """Return a clean in-place Sklansky adder modulo ``2**width``."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    source = tuple(range(width))
    target = tuple(range(width, 2 * width))
    work_start = 2 * width
    work_size = sklansky_inplace_workspace_size(width)
    circuit = QuantumCircuit(2 * width + work_size)
    _append_clean_sklansky_add(
        circuit,
        tuple((circuit.qubits[index] for index in source)),
        tuple((circuit.qubits[index] for index in target)),
        None,
        tuple((circuit.qubits[index] for index in range(work_start, work_start + work_size))),
    )
    circuit.name = f"sklansky_add_{width}"
    return circuit


@lru_cache(maxsize=None)
def sklansky_half_adder(width: int) -> QuantumCircuit:
    """Return a clean Sklansky adder retaining the carry-out."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    work_size = sklansky_inplace_workspace_size(width)
    circuit = QuantumCircuit(2 * width + 1 + work_size)
    _append_clean_sklansky_add(
        circuit,
        tuple(circuit.qubits[:width]),
        tuple(circuit.qubits[width : 2 * width]),
        circuit.qubits[2 * width],
        tuple(circuit.qubits[2 * width + 1 :]),
    )
    circuit.name = f"sklansky_half_add_{width}"
    return circuit


def append_sklansky_add(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    workspace: Sequence[Qubit],
    inverse: bool = False,
) -> None:
    """Append a clean Sklansky add/subtract modulo ``2**width``."""
    if len(source) != len(target) or not source:
        raise ValueError("source and target must have equal positive width")
    adder = sklansky_fixed_adder(len(source))
    operation = adder.inverse() if inverse else adder
    required = sklansky_inplace_workspace_size(len(source))
    if len(workspace) < required:
        raise ValueError("insufficient in-place Sklansky workspace")
    circuit.compose(operation, qubits=[*source, *target, *workspace[:required]], inplace=True)


def append_sklansky_sum(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    output: Sequence[Qubit],
    workspace: Sequence[Qubit],
) -> None:
    """Append ``output ^= left + right (mod 2**width)`` and clean work."""
    if len(left) != len(right) or len(left) != len(output) or (not left):
        raise ValueError("left, right, and output need equal positive width")
    required = sklansky_prefix_workspace_size(len(left))
    if len(workspace) < required:
        raise ValueError("insufficient out-of-place Sklansky workspace")
    _append_sklansky_sum_xor(
        circuit, left, right, output, workspace[:required], complement_right=False, carry_in=0
    )


def append_sklansky_difference(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    output: Sequence[Qubit],
    workspace: Sequence[Qubit],
) -> None:
    """Append ``output ^= left - right (mod 2**width)`` and clean work."""
    if len(left) != len(right) or len(left) != len(output) or (not left):
        raise ValueError("left, right, and output need equal positive width")
    required = sklansky_prefix_workspace_size(len(left))
    if len(workspace) < required:
        raise ValueError("insufficient out-of-place Sklansky workspace")
    _append_sklansky_sum_xor(
        circuit, left, right, output, workspace[:required], complement_right=True, carry_in=1
    )


def append_sklansky_half_add(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    carry: Qubit,
    workspace: Sequence[Qubit],
    inverse: bool = False,
) -> None:
    """Append a clean Sklansky half-adder or its inverse."""
    if len(source) != len(target) or not source:
        raise ValueError("source and target must have equal positive width")
    adder = sklansky_half_adder(len(source))
    operation = adder.inverse() if inverse else adder
    required = sklansky_inplace_workspace_size(len(source))
    if len(workspace) < required:
        raise ValueError("insufficient in-place Sklansky workspace")
    circuit.compose(
        operation, qubits=[*source, *target, carry, *workspace[:required]], inplace=True
    )


def append_sklansky_less_than(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    flag: Qubit,
    prefix_workspace: Sequence[Qubit],
) -> None:
    """Append ``flag ^= [left < right]`` with a Sklansky prefix tree."""
    left = tuple(left)
    right = tuple(right)
    if len(left) != len(right) or not left:
        raise ValueError("left and right must have equal positive width")
    _, prefixes_p, prefixes_g, operations = _append_sklansky_prefix_compute(
        circuit, left, right, prefix_workspace, complement_right=True
    )
    circuit.x(flag)
    circuit.cx(prefixes_g[-1], flag)
    circuit.cx(prefixes_p[-1], flag)
    _uncompute_recorded_gates(circuit, operations)
