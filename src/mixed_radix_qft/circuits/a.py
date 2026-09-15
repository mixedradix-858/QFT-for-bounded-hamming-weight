"""A circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.linear_mod import build_linear_mod_xor
from mixed_radix_qft.circuits.workspace import append_block, fresh
from mixed_radix_qft.registers import crt_layout


def build_general_a(moduli):
    """Map |r_j>|0_work> to |g_j*r_j mod m_j>|0_work> on each field.

    Here g_j=(m/m_j)^(-1) mod m_j and r_j<m_j. Fields are little-endian."""
    moduli = tuple(moduli)
    modulus, _, widths, _ = crt_layout(moduli)
    circuit = QuantumCircuit(sum(widths), name="general_A")
    offset = 0
    for p, width in zip(moduli, widths):
        field = circuit.qubits[offset : offset + width]
        offset += width
        result = fresh(circuit, width)
        inverse = pow(modulus // p, -1, p)
        append_block(
            circuit,
            build_linear_mod_xor([inverse * (1 << i) for i in range(width)], p),
            field + result,
        )
        append_block(
            circuit,
            build_linear_mod_xor([pow(inverse, -1, p) * (1 << i) for i in range(width)], p),
            result + field,
        )
        for left, right in zip(field, result):
            circuit.cx(left, right)
            circuit.cx(right, left)
            circuit.cx(left, right)
    circuit.metadata = {"moduli": list(moduli), "arithmetic": "wallace_qfa2_kogge_stone"}
    return circuit
