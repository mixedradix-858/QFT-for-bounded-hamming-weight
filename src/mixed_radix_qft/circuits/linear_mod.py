"""Linear mod circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.circuits.fanout import fanout
from mixed_radix_qft.circuits.linear_sum import raw_weighted_sum
from mixed_radix_qft.circuits.workspace import fresh, undo


def build_linear_mod_xor(coefficients, modulus):
    """Return |b>|z>|0> -> |b>|z XOR (sum c_i*b_i mod m)>|0>."""
    if not isinstance(modulus, int) or isinstance(modulus, bool) or modulus < 2:
        raise ValueError("modulus must be an integer >=2")
    if not coefficients or any((not isinstance(c, int) for c in coefficients)):
        raise ValueError("nonempty integer coefficient sequence required")
    coefficients = tuple((c % modulus for c in coefficients))
    count, output_width = (len(coefficients), (modulus - 1).bit_length())
    circuit = QuantumCircuit(count + output_width, name=f"parallel_mod_{modulus}")
    inputs, output = (circuit.qubits[:count], circuit.qubits[count:])
    maximum = sum(coefficients)
    width = max(output_width, maximum.bit_length())
    start = len(circuit.data)
    total = raw_weighted_sum(circuit, inputs, coefficients, width)
    quotient_max = maximum // modulus
    if quotient_max:
        copies = [fresh(circuit, width) for _ in range(quotient_max)]
        for bit in range(width):
            fanout(circuit, total[bit], [word[bit] for word in copies])
        flags = fresh(circuit, quotient_max)
        for multiple, (copy, flag) in enumerate(zip(copies, flags), 1):
            append_log_depth_constant_comparator(
                circuit,
                copy,
                flag,
                multiple * modulus,
                fresh(circuit, comparator_workspace_size(width)),
            )
        total = raw_weighted_sum(
            circuit,
            total + flags,
            [1 << bit for bit in range(width)] + [-modulus] * quotient_max,
            width,
        )
    stop = len(circuit.data)
    for source, target in zip(total, output):
        circuit.cx(source, target)
    undo(circuit, start, stop)
    circuit.metadata = {
        "arithmetic": "wallace_qfa2_kogge_stone",
        "modulus": modulus,
        "public_qubits": count + output_width,
    }
    return circuit
