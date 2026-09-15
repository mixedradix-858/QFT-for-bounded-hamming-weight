"""Weighted sum circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.dispatch import _append_modular_sum
from mixed_radix_qft.circuits.adders.qfa2 import append_qfa2_word_compressor
from mixed_radix_qft.circuits.adders.sklansky import (
    append_sklansky_sum,
    sklansky_prefix_workspace_size,
)


def _append_balanced_sum_tree(
    circuit: QuantumCircuit,
    sources: Sequence[Sequence[Qubit]],
    label: str,
    explicit_primitives: bool = False,
) -> tuple[Qubit, ...]:
    """Append the legacy Cuccaro implementation of ``W_j``."""
    if not sources:
        raise ValueError("the sum tree needs at least one source")
    width = len(sources[0])
    current = [tuple(source) for source in sources]
    level = 0
    while len(current) > 1:
        next_level: list[tuple[Qubit, ...]] = []
        parent_index = 0
        for cursor in range(0, len(current), 2):
            if cursor + 1 == len(current):
                next_level.append(current[cursor])
                continue
            parent = QuantumRegister(width, f"{label}_sum_l{level}_{parent_index}")
            circuit.add_register(parent)
            parent_qubits = tuple(parent)
            helper: Qubit | None = None
            if explicit_primitives:
                helper_register = QuantumRegister(1, f"{label}_sum_l{level}_{parent_index}_carry")
                circuit.add_register(helper_register)
                helper = helper_register[0]
            _append_modular_sum(circuit, current[cursor], parent_qubits, "W_left", helper)
            _append_modular_sum(circuit, current[cursor + 1], parent_qubits, "W_right", helper)
            next_level.append(parent_qubits)
            parent_index += 1
        current = next_level
        level += 1
    return current[0]


def _append_wallace_qfa2_sum(
    circuit: QuantumCircuit, sources: Sequence[Sequence[Qubit]], label: str
) -> tuple[Qubit, ...]:
    """Compute sum_c P_c using QFA2 compressors and a final Sklansky sum.

    The width must fit the promised sum. Retain intermediate words for inversion."""
    if not sources:
        raise ValueError("the Wallace sum needs at least one source")
    width = len(sources[0])
    if width <= 0 or any((len(source) != width for source in sources)):
        raise ValueError("all Wallace operands need the same positive width")
    current = [tuple(source) for source in sources]
    level = 0
    while len(current) > 2:
        next_level: list[tuple[Qubit, ...]] = []
        compressor_index = 0
        cursor = 0
        while cursor + 2 < len(current):
            first, second, third = current[cursor : cursor + 3]
            carry = QuantumRegister(width, f"{label}_carry_l{level}_{compressor_index}")
            circuit.add_register(carry)
            carry_qubits = tuple(carry)
            append_qfa2_word_compressor(circuit, first, second, third, carry_qubits)
            next_level.extend((third, carry_qubits))
            compressor_index += 1
            cursor += 3
        next_level.extend(current[cursor:])
        current = next_level
        level += 1
    if len(current) == 1:
        return current[0]
    output = QuantumRegister(width, f"{label}_sum")
    prefix = QuantumRegister(sklansky_prefix_workspace_size(width), f"{label}_prefix")
    circuit.add_register(output)
    circuit.add_register(prefix)
    append_sklansky_sum(circuit, current[0], current[1], output, prefix)
    return tuple(output)


def build_wallace_qfa2_sum_circuit(
    num_operands: int, width: int
) -> tuple[QuantumCircuit, tuple[int, ...]]:
    """Compute sum(inputs) modulo 2**width using reversible QFA2 compression.

    Return the circuit and sum-wire indices. Operands can be overwritten;
    intermediate history remains live until the complete block is reversed.
    """
    if num_operands <= 0 or width <= 0:
        raise ValueError("num_operands and width must be positive")
    inputs = QuantumRegister(num_operands * width, "operands")
    circuit = QuantumCircuit(inputs, name=f"wallace_qfa2_{num_operands}x{width}")
    sources = tuple(
        (
            tuple((inputs[operand * width + bit] for bit in range(width)))
            for operand in range(num_operands)
        )
    )
    output = _append_wallace_qfa2_sum(circuit, sources, "wallace")
    output_indices = tuple((circuit.find_bit(qubit).index for qubit in output))
    return (circuit, output_indices)


def wallace_qfa2_reduction_shape(num_operands: int) -> tuple[int, int]:
    """Return ``(levels, compressors)`` for a greedy Wallace 3:2 tree."""
    if num_operands <= 0:
        raise ValueError("a Wallace tree needs at least one operand")
    remaining = num_operands
    levels = 0
    compressors = 0
    while remaining > 2:
        groups, remainder = divmod(remaining, 3)
        compressors += groups
        remaining = 2 * groups + remainder
        levels += 1
    return (levels, compressors)
