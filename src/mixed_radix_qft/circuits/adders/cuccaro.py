"""Cuccaro circuits and mathematical helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.synthesis.arithmetic.adders import adder_ripple_c04

from mixed_radix_qft.circuits.controls import append_log_depth_mcx


@lru_cache(maxsize=None)
def cuccaro_fixed_adder(width: int) -> QuantumCircuit:
    """Return |a>|b>|0> -> |a>|a+b mod 2**width>|0> in CX/CCX."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    adder = adder_ripple_c04(width, kind="fixed")
    expanded = adder.decompose(reps=1)
    unexpected = set(expanded.count_ops()) - {"cx", "ccx"}
    if unexpected:
        raise AssertionError(f"unexpected gates in expanded Cuccaro adder: {sorted(unexpected)}")
    expanded.name = f"cuccaro_add_{width}"
    return expanded


@lru_cache(maxsize=None)
def cuccaro_half_adder(width: int) -> QuantumCircuit:
    """Return ``|a>|b>|0>|0> -> |a>|a+b>|carry>|0>``."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    adder = adder_ripple_c04(width, kind="half")
    expanded = adder.decompose(reps=1)
    unexpected = set(expanded.count_ops()) - {"cx", "ccx"}
    if unexpected:
        raise AssertionError(
            f"unexpected gates in expanded Cuccaro half adder: {sorted(unexpected)}"
        )
    expanded.name = f"cuccaro_half_add_{width}"
    return expanded


@lru_cache(maxsize=None)
def cuccaro_full_adder(width: int) -> QuantumCircuit:
    """Return the Cuccaro full adder with explicit carry-in/out."""
    if width <= 0:
        raise ValueError("adder width must be positive")
    adder = adder_ripple_c04(width, kind="full")
    expanded = adder.decompose(reps=1)
    unexpected = set(expanded.count_ops()) - {"cx", "ccx"}
    if unexpected:
        raise AssertionError(
            f"unexpected gates in expanded Cuccaro full adder: {sorted(unexpected)}"
        )
    expanded.name = f"cuccaro_full_add_{width}"
    return expanded


def append_cuccaro_add(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    helper: Qubit,
    inverse: bool = False,
) -> None:
    """Append an explicit Cuccaro add or subtract modulo 2**b."""
    if len(source) != len(target) or not source:
        raise ValueError("source and target must have the same positive width")
    adder = cuccaro_fixed_adder(len(source))
    operation = adder.inverse() if inverse else adder
    circuit.compose(operation, qubits=[*source, *target, helper], inplace=True)


def append_cuccaro_half_add(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    carry: Qubit,
    helper: Qubit,
    inverse: bool = False,
) -> None:
    """Append an exact addition retaining the carry-out."""
    if len(source) != len(target) or not source:
        raise ValueError("source and target must have the same positive width")
    adder = cuccaro_half_adder(len(source))
    operation = adder.inverse() if inverse else adder
    circuit.compose(operation, qubits=[*source, *target, carry, helper], inplace=True)


def append_cuccaro_full_add(
    circuit: QuantumCircuit,
    carry_in: Qubit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    carry_out: Qubit,
    inverse: bool = False,
) -> None:
    """Append the exact Cuccaro full adder or its inverse."""
    if len(source) != len(target) or not source:
        raise ValueError("source and target must have the same positive width")
    adder = cuccaro_full_adder(len(source))
    operation = adder.inverse() if inverse else adder
    circuit.compose(operation, qubits=[carry_in, *source, *target, carry_out], inplace=True)


def append_controlled_cuccaro_add(
    circuit: QuantumCircuit,
    control: Qubit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    helper: Qubit,
    mcx_workspace: Sequence[Qubit],
    inverse: bool = False,
) -> None:
    """Control every CX/CCX in an expanded Cuccaro ripple-carry adder."""
    if len(source) != len(target) or not source:
        raise ValueError("source and target must have the same positive width")
    if len(mcx_workspace) < 2:
        raise ValueError("controlled Cuccaro addition needs two clean MCX qubits")
    base = cuccaro_fixed_adder(len(source))
    mapping = [*source, *target, helper]
    instructions = tuple(base.data)
    if inverse:
        instructions = tuple(reversed(instructions))
    for instruction in instructions:
        operation = instruction.operation
        mapped = [mapping[base.find_bit(qubit).index] for qubit in instruction.qubits]
        if operation.name == "cx":
            circuit.ccx(control, mapped[0], mapped[1])
        elif operation.name == "ccx":
            append_log_depth_mcx(circuit, [control, mapped[0], mapped[1]], mapped[2], mcx_workspace)
        else:
            raise AssertionError(f"unsupported Cuccaro primitive {operation.name!r}")
