"""General circuits and mathematical helpers."""

from __future__ import annotations

from mixed_radix_qft.circuits.a import build_general_a
from mixed_radix_qft.circuits.c import build_general_c
from mixed_radix_qft.circuits.local_qft import build_mosca_zalka_inplace_qft
from mixed_radix_qft.circuits.workspace import append_block
from mixed_radix_qft.classical.number_theory import is_prime
from mixed_radix_qft.registers import crt_layout


def build_general_mixed_qft(moduli, *, coherent=True, local_backend="explicit-register-sklansky"):
    """Build C† (tensor_j F_mj) A C, with positive Fourier phase.

    For x<m, coherent output is on X and all other wires end zero.
    Terminal output is in ordinary CRT fields and requires classical decoding.
    The explicit local construction requires prime factors."""
    moduli = tuple(moduli)
    _, n, widths, fields = crt_layout(moduli)
    if any((not is_prime(p) for p in moduli)):
        raise ValueError("the selected MZ local backend requires prime factors")
    if local_backend not in {
        "explicit",
        "explicit-fourier",
        "explicit-fourier-history",
        "explicit-register",
        "explicit-register-sklansky",
    }:
        raise ValueError("an explicit MZ local backend is required")
    conversion = build_general_c(moduli)
    circuit = conversion.copy(
        name="general_mixed_coherent" if coherent else "general_mixed_terminal"
    )
    conversion_wires = list(circuit.qubits)
    residues = [circuit.qubits[i] for field in fields for i in field]
    append_block(circuit, build_general_a(moduli), residues)
    for p, field in zip(moduli, fields):
        target = [circuit.qubits[i] for i in field]
        if p == 2:
            circuit.h(target[0])
        else:
            append_block(
                circuit, build_mosca_zalka_inplace_qft(p, arithmetic=local_backend), target
            )
    if coherent:
        circuit.compose(conversion.inverse(), qubits=conversion_wires, inplace=True)
    circuit.metadata.update(
        {
            "coherent": coherent,
            "local_backend": local_backend,
            "output_wires": list(range(n)) if coherent else [i for f in fields for i in f],
        }
    )
    return circuit
