"""C circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.fanout import fanout
from mixed_radix_qft.circuits.linear_mod import build_linear_mod_xor
from mixed_radix_qft.circuits.workspace import append_block, fresh, undo
from mixed_radix_qft.registers import crt_layout


def build_general_c(moduli):
    """Map |x>|0_Y>|0_work> to |0_X>|(x mod m_j)_j>|0_work>.

    Requires 0 <= x < m and pairwise coprime factors. X and residue fields
    are little-endian. The inverse accepts every valid residue tuple."""
    moduli = tuple(moduli)
    modulus, n, widths, fields = crt_layout(moduli)
    circuit = QuantumCircuit(n + sum(widths), name="general_C")
    source = circuit.qubits[:n]
    copies = [fresh(circuit, n) for _ in moduli]
    start = len(circuit.data)
    for bit in range(n):
        fanout(circuit, source[bit], [word[bit] for word in copies])
    stop = len(circuit.data)
    for p, field, copy in zip(moduli, fields, copies):
        append_block(
            circuit,
            build_linear_mod_xor([1 << i for i in range(n)], p),
            copy + [circuit.qubits[i] for i in field],
        )
    undo(circuit, start, stop)
    coefficients = []
    for p, width in zip(moduli, widths):
        factor = modulus // p
        inverse = pow(factor, -1, p)
        coefficients.extend((factor * inverse * (1 << bit) for bit in range(width)))
    residues = [circuit.qubits[i] for field in fields for i in field]
    append_block(circuit, build_linear_mod_xor(coefficients, modulus), residues + source)
    circuit.metadata = {
        "moduli": list(moduli),
        "n": n,
        "fields": [list(f) for f in fields],
        "arithmetic": "wallace_qfa2_kogge_stone",
        "hamming_promise": False,
    }
    return circuit
