"""Range preparation circuits and mathematical helpers."""

from __future__ import annotations

from math import pi, sqrt
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.controls import (
    append_log_depth_controlled_ry,
    controlled_ry_workspace_size,
)
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


def _prefix_ctrl_state(prefix: int, prefix_width: int) -> int:
    """Encode conventional MSB-first prefix bits for a Qiskit control list."""
    state = 0
    for control_index in range(prefix_width):
        prefix_bit = prefix >> prefix_width - control_index - 1 & 1
        state |= prefix_bit << control_index
    return state


def range_preparation_workspace_size(prime: int) -> int:
    """Return conjunction scratch for the interval-state preparation."""
    width = ceil_log2(prime)
    return controlled_ry_workspace_size(max(0, width - 1))


def append_range_preparation(
    circuit: QuantumCircuit,
    target: Sequence[Qubit],
    workspace: Sequence[Qubit],
    prime: int,
    inverse: bool = False,
) -> None:
    """Prepare (1/sqrt(p))*sum_{y<p}|y> from a zero target.

    The inverse unprepares that state. Conjunction scratch starts and ends zero."""
    width = len(target)
    if width != ceil_log2(prime):
        raise ValueError("target width does not match the prime")
    if len(workspace) < range_preparation_workspace_size(prime):
        raise ValueError("insufficient clean workspace for range preparation")
    steps: list[tuple[Qubit, tuple[Qubit, ...], int, float]] = []
    previous_controls: list[Qubit] = []
    for depth, bit_index in enumerate(reversed(range(width))):
        target_qubit = target[bit_index]
        capacity = 1 << width - depth
        child_capacity = capacity // 2
        boundary_prefix: int | None = None
        boundary_population = 0
        prefix, population = divmod(prime, capacity)
        if population:
            boundary_prefix = prefix
            boundary_population = population
        steps.append((target_qubit, tuple(), 0, pi / 2))
        if boundary_prefix is not None:
            start = boundary_prefix * capacity
            zero_count = max(0, min(prime, start + child_capacity) - start)
            one_count = boundary_population - zero_count
            desired_angle = 2 * np.arctan2(sqrt(one_count), sqrt(zero_count))
            correction = float(desired_angle - pi / 2)
            controls = tuple(previous_controls)
            control_state = _prefix_ctrl_state(boundary_prefix, depth)
            if abs(correction) > 1e-15:
                steps.append((target_qubit, controls, control_state, correction))
        previous_controls.append(target_qubit)
    iterable = reversed(steps) if inverse else steps
    for target_qubit, controls, control_state, angle in iterable:
        append_log_depth_controlled_ry(
            circuit,
            -angle if inverse else angle,
            controls,
            target_qubit,
            workspace,
            ctrl_state=control_state,
        )


def build_range_preparation(prime: int) -> QuantumCircuit:
    """Map |0>|0_work> to sum_{y<p}|y>/sqrt(p) tensor |0_work>.

    The target has ceil(log2(p)) little-endian qubits.
    """
    if not is_prime(prime):
        raise ValueError("prime must be prime")
    width = ceil_log2(prime)
    target = QuantumRegister(width, "value")
    circuit = QuantumCircuit(target, name=f"RangePrep_{prime}")
    work_width = range_preparation_workspace_size(prime)
    if work_width:
        work = QuantumRegister(work_width, "range_work")
        circuit.add_register(work)
    else:
        work = ()
    append_range_preparation(circuit, target, work, prime)
    return circuit
