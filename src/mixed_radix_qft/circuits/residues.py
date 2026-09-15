"""Residues circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.bounded_reduction import validate_reduction_options
from mixed_radix_qft.circuits.contributions import _append_contribution_lookups
from mixed_radix_qft.circuits.counters import _append_counter_forest
from mixed_radix_qft.circuits.fanout import _fanout_to_targets, _uncompute_fanout
from mixed_radix_qft.circuits.modular_reduction import _append_threshold_reduction
from mixed_radix_qft.circuits.weighted_sum import (
    _append_balanced_sum_tree,
    _append_wallace_qfa2_sum,
)
from mixed_radix_qft.classical.crt import good_thomas_multipliers
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime
from mixed_radix_qft.classical.sparse_domain import periodic_classes
from mixed_radix_qft.config import WEIGHTED_SUM_BACKENDS, SparseQFTConfig


@dataclass(frozen=True)
class ResidueComputeBlock:
    circuit: QuantumCircuit
    temporary_indices: tuple[int, ...]
    periodic_class_count: int


def build_residue_compute_block(
    n: int,
    modulus: int,
    multiplier: int,
    w: int,
    label: str = "module",
    explicit_primitives: bool = False,
    weighted_sum_backend: str = "wallace-qfa2",
    *,
    reduction_backend: str = "cuccaro",
    reduction_cleanup: str = "local",
) -> ResidueComputeBlock:
    """Build U_j^(A)=R_j W_j L_j^(A) T_j, preserving X.

    For Ham(x)<=w, the temporary field holds g_j*x mod m_j.
    Intermediate counts and sums remain live until U_j^(A) is reversed."""
    validate_reduction_options(reduction_backend, reduction_cleanup, explicit_primitives)
    if n <= 0:
        raise ValueError("n must be positive")
    if not is_prime(modulus):
        raise ValueError("modulus must be prime")
    if w < 0 or w > n:
        raise ValueError("w must satisfy 0 <= w <= n")
    if gcd(multiplier, modulus) != 1:
        raise ValueError("multiplier must be invertible modulo modulus")
    if weighted_sum_backend not in WEIGHTED_SUM_BACKENDS:
        raise ValueError("weighted_sum_backend must be wallace-qfa2 or cuccaro")
    x_register = QuantumRegister(n, "x")
    circuit = QuantumCircuit(x_register, name=f"U_A_{modulus}")
    if w == 0:
        temporary = QuantumRegister(ceil_log2(modulus), f"{label}_tmp")
        circuit.add_register(temporary)
        temporary_indices = tuple((circuit.find_bit(qubit).index for qubit in temporary))
        return ResidueComputeBlock(circuit, temporary_indices, len(periodic_classes(n, modulus)))
    if modulus == 2:
        temporary = QuantumRegister(1, f"{label}_tmp")
        circuit.add_register(temporary)
        circuit.cx(x_register[0], temporary[0])
        temporary_indices = (circuit.find_bit(temporary[0]).index,)
        return ResidueComputeBlock(circuit, temporary_indices, 1)
    roots = _append_counter_forest(
        circuit, x_register, modulus, w, f"{label}_T", explicit_primitives
    )
    contributions = _append_contribution_lookups(
        circuit, roots, modulus, multiplier, w, f"{label}_L", explicit_primitives
    )
    if weighted_sum_backend == "wallace-qfa2":
        weighted_sum = _append_wallace_qfa2_sum(circuit, contributions, f"{label}_W")
    else:
        weighted_sum = _append_balanced_sum_tree(
            circuit, contributions, f"{label}_W", explicit_primitives
        )
    temporary = _append_threshold_reduction(
        circuit,
        weighted_sum,
        modulus,
        w,
        f"{label}_R",
        explicit_primitives,
        reduction_backend=reduction_backend,
        reduction_cleanup=reduction_cleanup,
    )
    temporary_indices = tuple((circuit.find_bit(qubit).index for qubit in temporary))
    return ResidueComputeBlock(circuit, temporary_indices, len(roots))


def _append_compute_copy_uncompute(
    circuit: QuantumCircuit,
    source_x: Sequence[Qubit],
    output: Sequence[Qubit],
    block: ResidueComputeBlock,
    label: str,
) -> int:
    """Append U_j, copy the temporary residue to clean Y_j, then append U_j†.

    X is preserved; every U_j workspace wire returns to zero on promised input."""
    n = len(source_x)
    work_width = block.circuit.num_qubits - n
    work = QuantumRegister(work_width, f"{label}_work")
    circuit.add_register(work)
    mapping = [*source_x, *work]
    circuit.compose(block.circuit, qubits=mapping, inplace=True)
    temporary = [mapping[index] for index in block.temporary_indices]
    if len(temporary) != len(output):
        raise ValueError("temporary and output widths must match")
    for source, target in zip(temporary, output, strict=True):
        circuit.cx(source, target)
    circuit.compose(block.circuit.inverse(), qubits=mapping, inplace=True)
    return work_width


def build_architecture_c_res(
    config: SparseQFTConfig,
    explicit_primitives: bool = False,
    weighted_sum_backend: str = "wallace-qfa2",
    *,
    reduction_backend: str = "cuccaro",
    reduction_cleanup: str = "local",
) -> QuantumCircuit:
    """Map |x>|0_Y>|0_work> to |x>|eta(x)>|0_work>.

    Requires x<m and Ham(x)<=w; eta_j(x)=g_j*x mod m_j.
    All fields are little-endian; X is preserved."""
    config.validate()
    x_register = QuantumRegister(config.n, "x")
    y_register = QuantumRegister(config.q_y, "y")
    circuit = QuantumCircuit(x_register, y_register, name="C_res_A_architecture")
    x_sources: list[Sequence[Qubit]] = [tuple(x_register)]
    for module_index in range(1, len(config.moduli)):
        copy_register = QuantumRegister(config.n, f"x_copy_{module_index}")
        circuit.add_register(copy_register)
        x_sources.append(tuple(copy_register))
    fanout_operations: list[list[tuple[Qubit, Qubit]]] = []
    for bit in range(config.n):
        targets = [source[bit] for source in x_sources[1:]]
        fanout_operations.append(_fanout_to_targets(circuit, x_register[bit], targets))
    offset = 0
    multipliers = good_thomas_multipliers(config.moduli)
    for module_index, (modulus, width, multiplier) in enumerate(
        zip(config.moduli, config.widths, multipliers, strict=True)
    ):
        output = tuple((y_register[offset + bit] for bit in range(width)))
        block = build_residue_compute_block(
            config.n,
            modulus,
            multiplier,
            config.w,
            label=f"m{module_index}_{modulus}",
            explicit_primitives=explicit_primitives,
            weighted_sum_backend=weighted_sum_backend,
            reduction_backend=reduction_backend,
            reduction_cleanup=reduction_cleanup,
        )
        _append_compute_copy_uncompute(
            circuit, x_sources[module_index], output, block, f"m{module_index}_{modulus}"
        )
        offset += width
    for operations in reversed(fanout_operations):
        _uncompute_fanout(circuit, operations)
    return circuit
