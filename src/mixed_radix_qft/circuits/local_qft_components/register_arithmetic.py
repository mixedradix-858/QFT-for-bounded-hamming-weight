"""Register arithmetic circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.cuccaro import (
    append_cuccaro_add,
    append_cuccaro_full_add,
    append_cuccaro_half_add,
)
from mixed_radix_qft.circuits.adders.fourier import append_fourier_controlled_constant_additions
from mixed_radix_qft.circuits.adders.sklansky import (
    append_sklansky_add,
    append_sklansky_half_add,
    append_sklansky_less_than,
    sklansky_inplace_workspace_size,
    sklansky_prefix_workspace_size,
)
from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


@dataclass(frozen=True)
class RegisterModularWorkspace:
    adder: str
    normalize: Qubit
    condition: Qubit
    sum_carry: Qubit
    add_helper: Qubit
    compare_cin: Qubit
    compare_cout: Qubit
    comparator: tuple[Qubit, ...]
    sklansky: tuple[Qubit, ...]


def register_modular_workspace_size(width: int, adder: str = "cuccaro") -> int:
    """Count scratch for clean modular register arithmetic with the chosen adder."""
    if width <= 0:
        raise ValueError("width must be positive")
    if adder not in {"cuccaro", "sklansky"}:
        raise ValueError("adder must be cuccaro or sklansky")
    shared = comparator_workspace_size(width)
    if adder == "sklansky":
        shared = max(shared, sklansky_inplace_workspace_size(width))
    return 6 + shared


def _partition_register_modular_workspace(
    qubits: Sequence[Qubit], width: int, adder: str = "cuccaro"
) -> RegisterModularWorkspace:
    """Assign non-overlapping predicate bits and reusable arithmetic scratch."""
    required = register_modular_workspace_size(width, adder)
    if len(qubits) < required:
        raise ValueError("insufficient register-modular workspace")
    comparator_size = comparator_workspace_size(width)
    sklansky_size = sklansky_inplace_workspace_size(width) if adder == "sklansky" else 0
    shared = tuple(qubits[6:required])
    return RegisterModularWorkspace(
        adder=adder,
        normalize=qubits[0],
        condition=qubits[1],
        sum_carry=qubits[2],
        add_helper=qubits[3],
        compare_cin=qubits[4],
        compare_cout=qubits[5],
        comparator=shared[:comparator_size],
        sklansky=shared[:sklansky_size],
    )


def append_register_less_than(
    circuit: QuantumCircuit,
    left: Sequence[Qubit],
    right: Sequence[Qubit],
    flag: Qubit,
    carry_in: Qubit,
    carry_out: Qubit,
    *,
    adder: str = "cuccaro",
    sklansky_workspace: Sequence[Qubit] = (),
) -> None:
    """Append ``flag ^= [left < right]`` and restore both registers."""
    left = tuple(left)
    right = tuple(right)
    if len(left) != len(right) or not left:
        raise ValueError("left and right must have the same positive width")
    if adder == "sklansky":
        required = sklansky_prefix_workspace_size(len(left))
        if len(sklansky_workspace) < required:
            raise ValueError("insufficient Sklansky comparator workspace")
        append_sklansky_less_than(circuit, left, right, flag, sklansky_workspace[:required])
        return
    if adder != "cuccaro":
        raise ValueError("adder must be cuccaro or sklansky")
    for qubit in right:
        circuit.x(qubit)
    circuit.x(carry_in)
    append_cuccaro_full_add(circuit, carry_in, right, left, carry_out)
    circuit.x(carry_out)
    circuit.cx(carry_out, flag)
    circuit.x(carry_out)
    append_cuccaro_full_add(circuit, carry_in, right, left, carry_out, inverse=True)
    circuit.x(carry_in)
    for qubit in right:
        circuit.x(qubit)


def append_modular_register_add(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    prime: int,
    workspace: RegisterModularWorkspace,
) -> None:
    """Append ``target += source (mod prime)`` on valid prime labels."""
    source = tuple(source)
    target = tuple(target)
    width = len(target)
    if len(source) != width or width != ceil_log2(prime):
        raise ValueError("register widths do not match the prime")
    if workspace.adder == "sklansky":
        append_sklansky_half_add(circuit, source, target, workspace.condition, workspace.sklansky)
    else:
        append_cuccaro_half_add(circuit, source, target, workspace.condition, workspace.add_helper)
    append_log_depth_constant_comparator(
        circuit, target, workspace.condition, prime, workspace.comparator
    )
    append_fourier_controlled_constant_additions(circuit, (workspace.condition,), (-prime,), target)
    append_register_less_than(
        circuit,
        target,
        source,
        workspace.condition,
        workspace.compare_cin,
        workspace.compare_cout,
        adder=workspace.adder,
        sklansky_workspace=workspace.sklansky,
    )


def append_phase_modular_register_add(
    circuit: QuantumCircuit,
    phase: Sequence[Qubit],
    target: Sequence[Qubit],
    prime: int,
    workspace: RegisterModularWorkspace,
) -> None:
    """Append ``target += phase (mod prime)`` for any q-bit phase label."""
    phase = tuple(phase)
    target = tuple(target)
    if len(phase) != len(target) or len(phase) != ceil_log2(prime):
        raise ValueError("register widths do not match the prime")
    append_log_depth_constant_comparator(
        circuit, phase, workspace.normalize, prime, workspace.comparator
    )
    append_fourier_controlled_constant_additions(circuit, (workspace.normalize,), (-prime,), phase)
    append_modular_register_add(circuit, phase, target, prime, workspace)
    append_fourier_controlled_constant_additions(circuit, (workspace.normalize,), (prime,), phase)
    append_log_depth_constant_comparator(
        circuit, phase, workspace.normalize, prime, workspace.comparator
    )


def append_modular_register_subtract(
    circuit: QuantumCircuit,
    source: Sequence[Qubit],
    target: Sequence[Qubit],
    prime: int,
    workspace: RegisterModularWorkspace,
) -> None:
    """Append ``target -= source (mod prime)`` on valid prime labels."""
    source = tuple(source)
    target = tuple(target)
    width = len(target)
    if len(source) != width or width != ceil_log2(prime):
        raise ValueError("register widths do not match the prime")
    append_register_less_than(
        circuit,
        target,
        source,
        workspace.condition,
        workspace.compare_cin,
        workspace.compare_cout,
        adder=workspace.adder,
        sklansky_workspace=workspace.sklansky,
    )
    if workspace.adder == "sklansky":
        append_sklansky_add(circuit, source, target, workspace.sklansky, inverse=True)
    else:
        append_cuccaro_add(circuit, source, target, workspace.add_helper, inverse=True)
    append_fourier_controlled_constant_additions(circuit, (workspace.condition,), (prime,), target)
    if workspace.adder == "sklansky":
        append_sklansky_half_add(circuit, source, target, workspace.sum_carry, workspace.sklansky)
    else:
        append_cuccaro_half_add(circuit, source, target, workspace.sum_carry, workspace.add_helper)
    append_log_depth_constant_comparator(
        circuit, target, workspace.sum_carry, prime, workspace.comparator
    )
    circuit.cx(workspace.sum_carry, workspace.condition)
    append_log_depth_constant_comparator(
        circuit, target, workspace.sum_carry, prime, workspace.comparator
    )
    if workspace.adder == "sklansky":
        append_sklansky_half_add(
            circuit, source, target, workspace.sum_carry, workspace.sklansky, inverse=True
        )
    else:
        append_cuccaro_half_add(
            circuit, source, target, workspace.sum_carry, workspace.add_helper, inverse=True
        )


def build_modular_register_operation(
    prime: int, operation: str = "add", adder: str = "cuccaro"
) -> QuantumCircuit:
    """Map |a>|b>|0_work> to |a>|b +/- a mod p>|0_work>.

    Addition/subtraction requires a,b<p; phase-add permits any q-bit a.
    Register order is source, target, then clean arithmetic scratch.
    """
    if not is_prime(prime) or prime == 2:
        raise ValueError("use an odd prime")
    if operation not in {"add", "phase-add", "subtract"}:
        raise ValueError("operation must be add, phase-add, or subtract")
    width = ceil_log2(prime)
    source = QuantumRegister(width, "source")
    target = QuantumRegister(width, "target")
    work = QuantumRegister(register_modular_workspace_size(width, adder), "register_modular_work")
    circuit = QuantumCircuit(source, target, work, name=f"register_modular_{operation}_{prime}")
    workspace = _partition_register_modular_workspace(work, width, adder)
    if operation == "add":
        append_modular_register_add(circuit, source, target, prime, workspace)
    elif operation == "phase-add":
        append_phase_modular_register_add(circuit, source, target, prime, workspace)
    else:
        append_modular_register_subtract(circuit, source, target, prime, workspace)
    return circuit
