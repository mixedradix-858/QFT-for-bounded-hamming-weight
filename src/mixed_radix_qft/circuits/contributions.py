"""Contributions circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.controls import append_log_depth_mcx, mcx_workspace_size
from mixed_radix_qft.classical.number_theory import ceil_log2


def _append_contribution_lookups(
    circuit: QuantumCircuit,
    roots: Sequence[tuple[int, Sequence[Qubit]]],
    modulus: int,
    multiplier: int,
    w: int,
    label: str,
    explicit_primitives: bool = False,
) -> tuple[tuple[Qubit, ...], ...]:
    """Write P_c=((g_j*c) mod m_j)*N_c into clean contribution words.

    Counts satisfy 0<=N_c<=w and are preserved; invert to clear the words."""
    sum_width = max(1, ceil_log2(w * (modulus - 1) + 1))
    contributions: list[tuple[Qubit, ...]] = []
    for class_index, (residue, root) in enumerate(roots):
        contribution = QuantumRegister(sum_width, f"{label}_p{class_index}")
        circuit.add_register(contribution)
        lookup_workspace: Sequence[Qubit] = ()
        if explicit_primitives:
            workspace_width = mcx_workspace_size(len(root))
            if workspace_width:
                workspace_register = QuantumRegister(workspace_width, f"{label}_mcx{class_index}")
                circuit.add_register(workspace_register)
                lookup_workspace = tuple(workspace_register)
        coefficient = multiplier * residue % modulus
        for count_value in range(1, w + 1):
            value = coefficient * count_value
            for bit in range(sum_width):
                if value >> bit & 1:
                    if explicit_primitives:
                        append_log_depth_mcx(
                            circuit,
                            root,
                            contribution[bit],
                            lookup_workspace,
                            ctrl_state=count_value,
                        )
                    else:
                        circuit.mcx(list(root), contribution[bit], ctrl_state=count_value)
        contributions.append(tuple(contribution))
    return tuple(contributions)
