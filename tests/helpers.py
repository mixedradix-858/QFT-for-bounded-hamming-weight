"""Full physical-output checks adapted from the original validation suite."""

from math import sqrt

import numpy as np

from mixed_radix_qft.validation.fourier import independent_dft
from mixed_radix_qft.validation.simulation import simulate_classical_reversible_circuit


def basis_output(circuit, label):
    """Return the entire physical output integer of a reversible basis circuit."""
    bits = simulate_classical_reversible_circuit(
        circuit, [label >> i & 1 for i in range(label.bit_length())]
    )
    return sum((bit << index for index, bit in enumerate(bits)))


def error_gram(columns, order):
    """Compute coherent isometry operator error without a 2**qubits matrix."""
    reference = independent_dft(order)
    errors = []
    for x, column in enumerate(columns):
        error = column.copy()
        for y in range(order):
            error[y] = error.get(y, 0) - reference[y, x]
        errors.append(error)
    gram = np.array(
        [
            [
                sum((np.conj(a) * right.get(index, 0) for index, a in left.items()))
                for right in errors
            ]
            for left in errors
        ]
    )
    return sqrt(max(0.0, float(np.linalg.eigvalsh(gram)[-1])))
