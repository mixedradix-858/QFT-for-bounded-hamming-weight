"""Counters circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.dispatch import _append_modular_sum
from mixed_radix_qft.classical.number_theory import is_prime
from mixed_radix_qft.classical.sparse_domain import counter_width, periodic_classes


def _append_counter_forest(
    circuit: QuantumCircuit,
    x_register: Sequence[Qubit],
    modulus: int,
    w: int,
    label: str,
    explicit_primitives: bool = False,
) -> tuple[tuple[int, tuple[Qubit, ...]], ...]:
    """Compute N_c=sum_{i:2**i mod m_j=c} x_i in a balanced tree.

    Requires Ham(x)<=w. Preserve X and retain intermediate counts for inversion."""
    width = counter_width(w)
    roots: list[tuple[int, tuple[Qubit, ...]]] = []
    for residue, indices in periodic_classes(len(x_register), modulus):
        current: list[tuple[Qubit, ...]] = []
        for leaf_index, input_index in enumerate(indices):
            leaf = QuantumRegister(width, f"{label}_c{residue}_leaf{leaf_index}")
            circuit.add_register(leaf)
            circuit.cx(x_register[input_index], leaf[0])
            current.append(tuple(leaf))
        level = 0
        while len(current) > 1:
            next_level: list[tuple[Qubit, ...]] = []
            parent_index = 0
            for cursor in range(0, len(current), 2):
                if cursor + 1 == len(current):
                    next_level.append(current[cursor])
                    continue
                parent = QuantumRegister(width, f"{label}_c{residue}_l{level + 1}_{parent_index}")
                circuit.add_register(parent)
                parent_qubits = tuple(parent)
                helper: Qubit | None = None
                if explicit_primitives:
                    helper_register = QuantumRegister(
                        1, f"{label}_c{residue}_l{level + 1}_{parent_index}_carry"
                    )
                    circuit.add_register(helper_register)
                    helper = helper_register[0]
                _append_modular_sum(circuit, current[cursor], parent_qubits, f"T_{modulus}", helper)
                _append_modular_sum(
                    circuit, current[cursor + 1], parent_qubits, f"T_{modulus}", helper
                )
                next_level.append(parent_qubits)
                parent_index += 1
            current = next_level
            level += 1
        roots.append((residue, current[0]))
    return tuple(roots)


def build_counter_forest_circuit(
    n: int, modulus: int, w: int, *, explicit_primitives: bool = True
) -> tuple[QuantumCircuit, tuple[tuple[int, tuple[int, ...]], ...]]:
    """Compute the periodic counts N_c, preserving X and retaining tree history.

    Requires Ham(x)<=w. Returned root indices locate little-endian counts;
    invert the circuit to clear all intermediate words."""
    if n <= 0 or not is_prime(modulus) or modulus == 2:
        raise ValueError("use n > 0 and an odd prime modulus")
    if w < 0 or w > n:
        raise ValueError("w must satisfy 0 <= w <= n")
    x_register = QuantumRegister(n, "x")
    circuit = QuantumCircuit(x_register, name=f"T_{modulus}")
    roots = _append_counter_forest(circuit, x_register, modulus, w, "counter", explicit_primitives)
    root_indices = tuple(
        (
            (residue, tuple((circuit.find_bit(qubit).index for qubit in root)))
            for residue, root in roots
        )
    )
    return (circuit, root_indices)
