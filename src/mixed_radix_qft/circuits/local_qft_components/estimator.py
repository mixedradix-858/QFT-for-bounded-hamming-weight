"""Estimator circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, sqrt

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import QFTGate

from mixed_radix_qft.circuits.controls import mcx_workspace_size
from mixed_radix_qft.circuits.local_qft_components.constant_arithmetic import (
    ModularArithmeticWorkspace,
    _partition_modular_workspace,
    append_controlled_modular_constant_add,
    modular_arithmetic_workspace_size,
)
from mixed_radix_qft.circuits.local_qft_components.modes import (
    EXPLICIT_ARITHMETIC_MODES,
    _addition_backend,
    _register_adder,
    _uses_history_predicates,
    _uses_register_arithmetic,
)
from mixed_radix_qft.circuits.local_qft_components.qfs import append_rephase
from mixed_radix_qft.circuits.local_qft_components.range_preparation import (
    append_range_preparation,
    range_preparation_workspace_size,
)
from mixed_radix_qft.circuits.local_qft_components.register_arithmetic import (
    RegisterModularWorkspace,
    _partition_register_modular_workspace,
    append_modular_register_subtract,
    append_phase_modular_register_add,
    register_modular_workspace_size,
)
from mixed_radix_qft.circuits.local_qft_components.rounding import (
    RoundingWorkspace,
    _partition_rounding_workspace,
    append_explicit_rounding,
    rounding_workspace_size,
)
from mixed_radix_qft.classical.estimator_parameters import uniform_success_probability
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


@dataclass(frozen=True)
class EstimatorRegisters:
    eigen: QuantumRegister
    randomizer: QuantumRegister
    phase: QuantumRegister
    estimate: QuantumRegister
    accepted: Qubit
    throttle: Qubit
    range_work: tuple[Qubit, ...]
    arithmetic_work: tuple[Qubit, ...]
    reflection_work: tuple[Qubit, ...]
    arithmetic: str


def build_uniformized_estimator(
    prime: int, arithmetic: str = "explicit-register-sklansky"
) -> tuple[QuantumCircuit, EstimatorRegisters]:
    """Build the MZ estimator with good-subspace probability 1/4.

    Preserve the Fourier eigenstate; all estimator work starts zero and
    remains live for exact amplification and subsequent uncomputation."""
    if not is_prime(prime) or prime == 2:
        raise ValueError("the prototype estimator expects an odd prime")
    supported_arithmetic = {*EXPLICIT_ARITHMETIC_MODES}
    if arithmetic not in supported_arithmetic:
        raise ValueError(
            "arithmetic must be explicit, explicit-fourier, explicit-fourier-history, explicit-register, or explicit-register-sklansky"
        )
    addition = _addition_backend(arithmetic)
    use_history = _uses_history_predicates(arithmetic)
    register_arithmetic = _uses_register_arithmetic(arithmetic)
    register_adder = _register_adder(arithmetic)
    width = ceil_log2(prime)
    eigen = QuantumRegister(width, "eigen")
    randomizer = QuantumRegister(width, "random")
    phase = QuantumRegister(width, "phase")
    estimate = QuantumRegister(width, "estimate")
    flags = QuantumRegister(2, "flags")
    circuit = QuantumCircuit(eigen, randomizer, phase, estimate, flags, name=f"A_MZ_{prime}")
    work_width = range_preparation_workspace_size(prime)
    if work_width:
        range_work_register = QuantumRegister(work_width, "range_work")
        circuit.add_register(range_work_register)
        range_work = tuple(range_work_register)
    else:
        range_work = ()
    arithmetic_work: tuple[Qubit, ...] = ()
    reflection_work: tuple[Qubit, ...] = ()
    modular_workspace: ModularArithmeticWorkspace | None = None
    register_workspace: RegisterModularWorkspace | None = None
    rounding_workspace: RoundingWorkspace | None = None
    history_predicates: tuple[Qubit, ...] = ()
    modular_size = 0 if register_arithmetic else modular_arithmetic_workspace_size(width, addition)
    register_size = (
        register_modular_workspace_size(width, register_adder) if register_arithmetic else 0
    )
    rounding_size = rounding_workspace_size(width, addition)
    history_size = width if use_history else 0
    arithmetic_register = QuantumRegister(
        modular_size + register_size + rounding_size + history_size, "arithmetic_work"
    )
    circuit.add_register(arithmetic_register)
    arithmetic_work = tuple(arithmetic_register)
    if register_arithmetic:
        register_workspace = _partition_register_modular_workspace(
            arithmetic_work[:register_size], width, register_adder
        )
    else:
        modular_workspace = _partition_modular_workspace(
            arithmetic_work[:modular_size], width, addition
        )
    rounding_offset = modular_size + register_size
    rounding_workspace = _partition_rounding_workspace(
        arithmetic_work[rounding_offset : rounding_offset + rounding_size], width, addition
    )
    history_predicates = tuple(arithmetic_work[rounding_offset + rounding_size :])
    a_ancilla_count = (
        len(randomizer) + len(phase) + len(estimate) + 2 + len(range_work) + len(arithmetic_work)
    )
    reflection_size = mcx_workspace_size(a_ancilla_count - 1)
    if reflection_size:
        reflection_register = QuantumRegister(reflection_size, "reflection_work")
        circuit.add_register(reflection_register)
        reflection_work = tuple(reflection_register)
    append_range_preparation(circuit, randomizer, range_work, prime)
    append_rephase(circuit, randomizer, eigen, prime)
    for qubit in phase:
        circuit.h(qubit)
    if register_arithmetic:
        if register_workspace is None:
            raise AssertionError("missing register-modular workspace")
        append_phase_modular_register_add(circuit, phase, eigen, prime, register_workspace)
    else:
        for bit, phase_qubit in enumerate(phase):
            if modular_workspace is None:
                raise AssertionError("missing explicit modular workspace")
            append_controlled_modular_constant_add(
                circuit, phase_qubit, eigen, prime, 1 << bit, modular_workspace, addition=addition
            )
    circuit.compose(QFTGate(width).definition, qubits=phase, inplace=True)
    if rounding_workspace is None:
        raise AssertionError("missing explicit arithmetic workspace")
    append_explicit_rounding(
        circuit,
        phase,
        estimate,
        flags[0],
        prime,
        modular_workspace,
        rounding_workspace,
        addition=addition,
    )
    if register_arithmetic:
        if register_workspace is None:
            raise AssertionError("missing register-modular workspace")
        append_modular_register_subtract(circuit, randomizer, estimate, prime, register_workspace)
    else:
        if modular_workspace is None:
            raise AssertionError("missing explicit modular workspace")
        for bit, random_qubit in enumerate(randomizer):
            append_controlled_modular_constant_add(
                circuit,
                random_qubit,
                estimate,
                prime,
                -(1 << bit),
                modular_workspace,
                addition=addition,
                history_predicate=history_predicates[bit] if use_history else None,
            )
    append_rephase(circuit, randomizer, eigen, prime, inverse=True)
    average_success = uniform_success_probability(prime)
    throttle_probability = 1 / (4 * average_success)
    if throttle_probability > 1 + 1e-12:
        raise AssertionError("cannot reduce estimator success to 1/4")
    throttle_angle = 2 * asin(sqrt(min(1.0, throttle_probability)))
    circuit.ry(throttle_angle, flags[1])
    registers = EstimatorRegisters(
        eigen=eigen,
        randomizer=randomizer,
        phase=phase,
        estimate=estimate,
        accepted=flags[0],
        throttle=flags[1],
        range_work=range_work,
        arithmetic_work=arithmetic_work,
        reflection_work=reflection_work,
        arithmetic=arithmetic,
    )
    return (circuit, registers)
