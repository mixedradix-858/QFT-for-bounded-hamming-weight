"""Reflections circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.controls import append_log_depth_mcx
from mixed_radix_qft.circuits.local_qft_components.estimator import EstimatorRegisters
from mixed_radix_qft.circuits.local_qft_components.modes import _uses_explicit_arithmetic
from mixed_radix_qft.circuits.local_qft_components.qfs import append_rephase
from mixed_radix_qft.circuits.local_qft_components.range_preparation import append_range_preparation


def _append_multi_controlled_phase_flip(
    circuit: QuantumCircuit,
    controls: Sequence[Qubit],
    target: Qubit,
    workspace: Sequence[Qubit] = (),
    explicit: bool = False,
) -> None:
    """Apply a phase -1 when all controls and target are one."""
    controls = tuple(controls)
    if not controls:
        circuit.z(target)
        return
    circuit.h(target)
    if explicit:
        append_log_depth_mcx(circuit, controls, target, workspace)
    else:
        circuit.mcx(list(controls), target)
    circuit.h(target)


def append_good_phase_oracle(
    circuit: QuantumCircuit, registers: EstimatorRegisters, prime: int
) -> None:
    """Recognize a correct estimate using the untouched Fourier eigenstate."""
    append_rephase(circuit, registers.estimate, registers.eigen, prime, inverse=True)
    append_range_preparation(circuit, registers.eigen, registers.range_work, prime, inverse=True)
    for qubit in registers.eigen:
        circuit.x(qubit)
    _append_multi_controlled_phase_flip(
        circuit,
        [*registers.eigen, registers.accepted],
        registers.throttle,
        registers.reflection_work,
        explicit=_uses_explicit_arithmetic(registers.arithmetic),
    )
    for qubit in registers.eigen:
        circuit.x(qubit)
    append_range_preparation(circuit, registers.eigen, registers.range_work, prime)
    append_rephase(circuit, registers.estimate, registers.eigen, prime)


def append_zero_ancilla_phase_oracle(
    circuit: QuantumCircuit, registers: EstimatorRegisters
) -> None:
    """Apply phase -1 when every A-work register is in its initial |0> state."""
    ancillas = [
        *registers.randomizer,
        *registers.phase,
        *registers.estimate,
        registers.accepted,
        registers.throttle,
        *registers.range_work,
        *registers.arithmetic_work,
    ]
    target = registers.throttle
    controls = [qubit for qubit in ancillas if qubit != target]
    for qubit in ancillas:
        circuit.x(qubit)
    _append_multi_controlled_phase_flip(
        circuit,
        controls,
        target,
        registers.reflection_work,
        explicit=_uses_explicit_arithmetic(registers.arithmetic),
    )
    for qubit in ancillas:
        circuit.x(qubit)
