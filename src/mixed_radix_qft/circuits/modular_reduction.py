"""Modular reduction circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import IntegerComparatorGate, ModularAdderGate

from mixed_radix_qft.circuits.adders.cuccaro import append_controlled_cuccaro_add
from mixed_radix_qft.circuits.bounded_reduction import (
    append_bounded_reduction,
    validate_reduction_options,
)
from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


def _append_threshold_reduction(
    circuit: QuantumCircuit,
    source_sum: Sequence[Qubit],
    modulus: int,
    w: int,
    label: str,
    explicit_primitives: bool = False,
    *,
    reduction_backend: str = "cuccaro",
    reduction_cleanup: str = "local",
) -> tuple[Qubit, ...]:
    """Compute S mod m_j from 0<=S<=w*(m_j-1), preserving S.

    The returned residue starts zero; cleanup controls internal history retention."""
    validate_reduction_options(reduction_backend, reduction_cleanup, explicit_primitives)
    if (reduction_backend, reduction_cleanup) != ("cuccaro", "local"):
        return append_bounded_reduction(
            circuit, source_sum, modulus, w, backend=reduction_backend, cleanup=reduction_cleanup
        )
    width = len(source_sum)
    residue_width = ceil_log2(modulus)
    work = QuantumRegister(width, f"{label}_reduce_work")
    temporary = QuantumRegister(residue_width, f"{label}_tmp")
    circuit.add_register(work)
    circuit.add_register(temporary)
    for source, target in zip(source_sum, work, strict=True):
        circuit.cx(source, target)
    max_sum = w * (modulus - 1)
    max_quotient = max_sum // modulus
    flags: QuantumRegister | None = None
    comparators: list[IntegerComparatorGate] = []
    comparator_workspaces: list[tuple[Qubit, ...]] = []
    if max_quotient:
        flags = QuantumRegister(max_quotient, f"{label}_thresholds")
        constant = QuantumRegister(width, f"{label}_minus_m")
        circuit.add_register(flags)
        circuit.add_register(constant)
        for quotient in range(1, max_quotient + 1):
            if explicit_primitives:
                comparator_work = QuantumRegister(
                    comparator_workspace_size(width), f"{label}_cmp{quotient}_work"
                )
                circuit.add_register(comparator_work)
                append_log_depth_constant_comparator(
                    circuit, source_sum, flags[quotient - 1], quotient * modulus, comparator_work
                )
                comparator_workspaces.append(tuple(comparator_work))
            else:
                comparator = IntegerComparatorGate(
                    width, quotient * modulus, geq=True, label=f">={quotient}m"
                )
                circuit.append(comparator, [*source_sum, flags[quotient - 1]])
                comparators.append(comparator)
        negative_modulus = -modulus % (1 << width)
        for bit in range(width):
            if negative_modulus >> bit & 1:
                circuit.x(constant[bit])
        controlled_adder = None
        controlled_helper: QuantumRegister | None = None
        controlled_mcx_work: QuantumRegister | None = None
        if explicit_primitives:
            controlled_helper = QuantumRegister(1, f"{label}_sub_carry")
            controlled_mcx_work = QuantumRegister(2, f"{label}_sub_mcx")
            circuit.add_register(controlled_helper)
            circuit.add_register(controlled_mcx_work)
            for flag in flags:
                append_controlled_cuccaro_add(
                    circuit, flag, constant, work, controlled_helper[0], controlled_mcx_work
                )
        else:
            controlled_adder = ModularAdderGate(width, label=f"-{modulus}").control(1)
            for flag in flags:
                circuit.append(controlled_adder, [flag, *constant, *work])
        for bit in range(residue_width):
            circuit.cx(work[bit], temporary[bit])
        for flag in reversed(flags):
            if explicit_primitives:
                assert controlled_helper is not None
                assert controlled_mcx_work is not None
                append_controlled_cuccaro_add(
                    circuit,
                    flag,
                    constant,
                    work,
                    controlled_helper[0],
                    controlled_mcx_work,
                    inverse=True,
                )
            else:
                assert controlled_adder is not None
                circuit.append(controlled_adder.inverse(), [flag, *constant, *work])
        for bit in range(width):
            if negative_modulus >> bit & 1:
                circuit.x(constant[bit])
    else:
        for bit in range(residue_width):
            circuit.cx(work[bit], temporary[bit])
    for source, target in zip(source_sum, work, strict=True):
        circuit.cx(source, target)
    if flags is not None:
        if explicit_primitives:
            for quotient in reversed(range(1, max_quotient + 1)):
                append_log_depth_constant_comparator(
                    circuit,
                    source_sum,
                    flags[quotient - 1],
                    quotient * modulus,
                    comparator_workspaces[quotient - 1],
                )
        else:
            for flag, comparator in reversed(tuple(zip(flags, comparators, strict=True))):
                circuit.append(comparator, [*source_sum, flag])
    return tuple(temporary)


def build_threshold_reduction_circuit(
    modulus: int, w: int, explicit_primitives: bool = False, *, reduction_backend: str = "cuccaro"
) -> tuple[QuantumCircuit, tuple[int, ...]]:
    """Map |S>|0_result>|0_work> to |S>|S mod p>|0_work>.

    Requires prime p and 0<=S<=w*(p-1), w>=1. Return residue-wire indices.
    """
    validate_reduction_options(reduction_backend, "local", explicit_primitives)
    if not is_prime(modulus):
        raise ValueError("modulus must be prime")
    if w < 1:
        raise ValueError("w must be positive")
    width = ceil_log2(w * (modulus - 1) + 1)
    source_sum = QuantumRegister(width, "sum")
    circuit = QuantumCircuit(source_sum, name=f"R_{modulus}")
    temporary = _append_threshold_reduction(
        circuit,
        source_sum,
        modulus,
        w,
        "reduction",
        explicit_primitives,
        reduction_backend=reduction_backend,
    )
    temporary_indices = tuple((circuit.find_bit(qubit).index for qubit in temporary))
    return (circuit, temporary_indices)
