"""Rounding circuits and mathematical helpers."""

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
from mixed_radix_qft.circuits.controls import append_log_depth_mcx
from mixed_radix_qft.circuits.local_qft_components.constant_arithmetic import (
    ModularArithmeticWorkspace,
    _partition_modular_workspace,
    modular_arithmetic_workspace_size,
)
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


@dataclass(frozen=True)
class RoundingWorkspace:
    product: tuple[Qubit, ...]
    constant: tuple[Qubit, ...]
    add_helper: Qubit
    add_mcx: tuple[Qubit, ...]
    comparator: tuple[Qubit, ...]
    nonzero: Qubit


def rounding_workspace_size(width: int, addition: str = "cuccaro") -> int:
    """Count product, predicate and arithmetic scratch for reversible rounding."""
    product_width = 2 * width
    if addition == "fourier":
        return product_width + comparator_workspace_size(width) + 1
    if addition != "cuccaro":
        raise ValueError("addition must be cuccaro or fourier")
    return product_width + product_width + 1 + 2 + comparator_workspace_size(width) + 1


def _partition_rounding_workspace(
    qubits: Sequence[Qubit], width: int, addition: str = "cuccaro"
) -> RoundingWorkspace:
    """Assign the exact product, nonzero predicate and rounding scratch."""
    if len(qubits) < rounding_workspace_size(width, addition):
        raise ValueError("insufficient rounding workspace")
    product_width = 2 * width
    cursor = 0
    product = tuple(qubits[cursor : cursor + product_width])
    cursor += product_width
    if addition == "fourier":
        constant = ()
        add_mcx = ()
        add_helper: Qubit | None = None
    else:
        constant = tuple(qubits[cursor : cursor + product_width])
        cursor += product_width
        add_helper = qubits[cursor]
        cursor += 1
        add_mcx = tuple(qubits[cursor : cursor + 2])
        cursor += 2
    comparator_width = comparator_workspace_size(width)
    comparator = tuple(qubits[cursor : cursor + comparator_width])
    cursor += comparator_width
    nonzero = qubits[cursor]
    if add_helper is None:
        add_helper = nonzero
    return RoundingWorkspace(
        product=product,
        constant=constant,
        add_helper=add_helper,
        add_mcx=add_mcx,
        comparator=comparator,
        nonzero=nonzero,
    )


def append_explicit_rounding(
    circuit: QuantumCircuit,
    phase: Sequence[Qubit],
    estimate: Sequence[Qubit],
    accepted: Qubit,
    prime: int,
    modular_workspace: ModularArithmeticWorkspace | None,
    rounding_workspace: RoundingWorkspace,
    addition: str = "cuccaro",
) -> None:
    """Compute k=ceil(y*p/2**q), preserving y, with zero estimate/flag inputs.

    Accept when r=(y*p) mod 2**q is zero or r>=2**q-p+1.
    Product and nonzero-predicate history remain live until this block is reversed.
    """
    width = len(phase)
    if len(estimate) != width:
        raise ValueError("phase and estimate registers must have equal width")
    product_width = 2 * width
    order = 1 << width
    if addition not in {"cuccaro", "fourier"}:
        raise ValueError("addition must be cuccaro or fourier")
    if addition == "fourier":
        append_fourier_controlled_constant_additions(
            circuit,
            phase,
            tuple((prime << bit for bit in range(width))),
            rounding_workspace.product,
        )
    else:
        for bit, control in enumerate(phase):
            append_controlled_constant_add(
                circuit,
                control,
                prime << bit,
                rounding_workspace.constant,
                rounding_workspace.product,
                rounding_workspace.add_helper,
                rounding_workspace.add_mcx,
            )
    low = rounding_workspace.product[:width]
    high = rounding_workspace.product[width:product_width]
    for source, target in zip(high, estimate, strict=True):
        circuit.cx(source, target)
    circuit.x(rounding_workspace.nonzero)
    append_log_depth_mcx(
        circuit, low, rounding_workspace.nonzero, rounding_workspace.comparator, ctrl_state=0
    )
    if addition == "fourier":
        append_fourier_controlled_constant_additions(
            circuit, (rounding_workspace.nonzero,), (1,), estimate
        )
    else:
        if modular_workspace is None:
            raise ValueError("Cuccaro rounding needs modular workspace")
        append_controlled_constant_add(
            circuit,
            rounding_workspace.nonzero,
            1,
            modular_workspace.constant,
            estimate,
            modular_workspace.add_helper,
            modular_workspace.add_mcx,
        )
    append_log_depth_mcx(circuit, low, accepted, rounding_workspace.comparator, ctrl_state=0)
    append_log_depth_constant_comparator(
        circuit, low, accepted, order - prime + 1, rounding_workspace.comparator
    )


def build_explicit_rounding_circuit(prime: int, addition: str = "cuccaro") -> QuantumCircuit:
    """Compute k=ceil(y*p/2**q) and its acceptance flag from a preserved y.

    For odd prime p, q=ceil(log2(p)); estimate and flag start zero.
    The exact product and rounding predicate remain live for inversion.
    """
    if not is_prime(prime) or prime == 2:
        raise ValueError("use an odd prime")
    width = ceil_log2(prime)
    phase = QuantumRegister(width, "phase")
    estimate = QuantumRegister(width, "estimate")
    accepted = QuantumRegister(1, "accepted")
    modular_work = QuantumRegister(
        modular_arithmetic_workspace_size(width, addition), "modular_work"
    )
    rounding_work = QuantumRegister(rounding_workspace_size(width, addition), "rounding_work")
    circuit = QuantumCircuit(
        phase, estimate, accepted, modular_work, rounding_work, name=f"round_and_filter_{prime}"
    )
    append_explicit_rounding(
        circuit,
        phase,
        estimate,
        accepted[0],
        prime,
        _partition_modular_workspace(modular_work, width, addition),
        _partition_rounding_workspace(rounding_work, width, addition),
        addition=addition,
    )
    return circuit
