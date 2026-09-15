"""Constant arithmetic circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.constant import append_controlled_constant_add
from mixed_radix_qft.circuits.adders.fourier import append_fourier_controlled_constant_additions
from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


@dataclass(frozen=True)
class ModularArithmeticWorkspace:
    constant: tuple[Qubit, ...]
    add_helper: Qubit
    add_mcx: tuple[Qubit, ...]
    predicate: Qubit
    wrap: Qubit
    comparator: tuple[Qubit, ...]


def modular_arithmetic_workspace_size(width: int, addition: str = "cuccaro") -> int:
    """Count scratch for controlled modular constant addition."""
    if addition == "fourier":
        return 2 + comparator_workspace_size(width)
    if addition != "cuccaro":
        raise ValueError("addition must be cuccaro or fourier")
    return width + 1 + 2 + 1 + 1 + comparator_workspace_size(width)


def _partition_modular_workspace(
    qubits: Sequence[Qubit], width: int, addition: str = "cuccaro"
) -> ModularArithmeticWorkspace:
    """Assign constant-addition registers and modular reduction predicates."""
    if len(qubits) < modular_arithmetic_workspace_size(width, addition):
        raise ValueError("insufficient modular-arithmetic workspace")
    cursor = 0
    if addition == "fourier":
        constant = ()
        add_mcx = ()
        add_helper: Qubit | None = None
    else:
        constant = tuple(qubits[cursor : cursor + width])
        cursor += width
        add_helper = qubits[cursor]
        cursor += 1
        add_mcx = tuple(qubits[cursor : cursor + 2])
        cursor += 2
    predicate = qubits[cursor]
    cursor += 1
    wrap = qubits[cursor]
    cursor += 1
    if add_helper is None:
        add_helper = predicate
    comparator_width = comparator_workspace_size(width)
    comparator = tuple(qubits[cursor : cursor + comparator_width])
    return ModularArithmeticWorkspace(
        constant=constant,
        add_helper=add_helper,
        add_mcx=add_mcx,
        predicate=predicate,
        wrap=wrap,
        comparator=comparator,
    )


def append_controlled_modular_constant_add(
    circuit: QuantumCircuit,
    control: Qubit,
    target: Sequence[Qubit],
    prime: int,
    value: int,
    workspace: ModularArithmeticWorkspace,
    addition: str = "cuccaro",
    history_predicate: Qubit | None = None,
) -> None:
    """Append ``target += value (mod prime)`` under one control."""
    width = len(target)
    if width != ceil_log2(prime):
        raise ValueError("target width does not match the prime")
    if addition not in {"cuccaro", "fourier"}:
        raise ValueError("addition must be cuccaro or fourier")
    shift = value % prime
    if shift == 0:
        return
    predicate = workspace.predicate if history_predicate is None else history_predicate
    if predicate == workspace.wrap:
        raise ValueError("history predicate and wrap qubits must be distinct")
    append_log_depth_constant_comparator(
        circuit, target, predicate, prime - shift, workspace.comparator
    )
    circuit.ccx(control, predicate, workspace.wrap)
    if history_predicate is None:
        append_log_depth_constant_comparator(
            circuit, target, predicate, prime - shift, workspace.comparator
        )
    if addition == "fourier":
        append_fourier_controlled_constant_additions(
            circuit, (control, workspace.wrap), (shift, -prime), target
        )
    else:
        append_controlled_constant_add(
            circuit,
            control,
            shift,
            workspace.constant,
            target,
            workspace.add_helper,
            workspace.add_mcx,
        )
        append_controlled_constant_add(
            circuit,
            workspace.wrap,
            -prime,
            workspace.constant,
            target,
            workspace.add_helper,
            workspace.add_mcx,
        )
    if history_predicate is not None:
        circuit.ccx(control, predicate, workspace.wrap)
        return
    append_log_depth_constant_comparator(
        circuit, target, workspace.predicate, shift, workspace.comparator
    )
    circuit.x(workspace.predicate)
    circuit.ccx(control, workspace.predicate, workspace.wrap)
    circuit.x(workspace.predicate)
    append_log_depth_constant_comparator(
        circuit, target, workspace.predicate, shift, workspace.comparator
    )


def build_controlled_modular_constant_add(
    prime: int, value: int, addition: str = "cuccaro", preserve_predicate: bool = False
) -> QuantumCircuit:
    """Map |c>|x>|0_work> to |c>|x+c*value mod p>|0_work>.

    Requires odd prime p and x<p. Preserve the control and clear scratch.
    """
    if not is_prime(prime) or prime == 2:
        raise ValueError("use an odd prime")
    width = ceil_log2(prime)
    control = QuantumRegister(1, "control")
    target = QuantumRegister(width, "target")
    work = QuantumRegister(modular_arithmetic_workspace_size(width, addition), "modular_work")
    circuit = QuantumCircuit(control, target, work, name=f"controlled_add_{value}_mod_{prime}")
    workspace = _partition_modular_workspace(work, width, addition)
    append_controlled_modular_constant_add(
        circuit,
        control[0],
        target,
        prime,
        value,
        workspace,
        addition=addition,
        history_predicate=workspace.predicate if preserve_predicate else None,
    )
    return circuit
