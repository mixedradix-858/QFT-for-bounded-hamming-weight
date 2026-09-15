"""Bounded-Hamming-weight input conversion followed by local Fourier transforms."""

from __future__ import annotations

from numbers import Integral

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.c import build_general_c
from mixed_radix_qft.circuits.local_qft import build_mosca_zalka_inplace_qft
from mixed_radix_qft.circuits.local_qft_components.modes import EXPLICIT_ARITHMETIC_MODES
from mixed_radix_qft.circuits.sparse_conversion import build_architecture_fused_conversion
from mixed_radix_qft.circuits.workspace import append_block
from mixed_radix_qft.classical.sparse_domain import sparse_value_count
from mixed_radix_qft.config import SparseQFTConfig


def build_sparse_qft(
    config: SparseQFTConfig,
    *,
    coherent: bool = True,
    local_backend: str = "explicit-register-sklansky",
    weighted_sum_backend: str = "wallace-qfa2",
    decoder: str = "lookup",
    reduction_backend: str = "cuccaro",
    reduction_cleanup: str = "local",
    lookup_entry_limit: int = 100_000,
) -> QuantumCircuit:
    """Map |x>|0_work> to F_m|x>|0_work> for x<m and Ham(x)<=w.

    Terminal mode omits general C† and returns ordinary CRT fields instead.
    The sparse promise applies only to the input, including superpositions.
    """
    config.validate()
    if config.n != config.canonical_n:
        raise ValueError("the article construction requires n=ceil(log2(m))")
    if local_backend not in EXPLICIT_ARITHMETIC_MODES:
        raise ValueError("an explicit local QFT synthesis is required")
    if (
        not isinstance(lookup_entry_limit, Integral)
        or isinstance(lookup_entry_limit, bool)
        or lookup_entry_limit < 1
    ):
        raise ValueError("lookup_entry_limit must be a positive integer")
    if decoder in {"lookup", "lookup-projected"}:
        count = sparse_value_count(config.n, config.w, config.crt_modulus)
        if count > lookup_entry_limit:
            raise MemoryError(f"input decoder needs {count} entries; limit is {lookup_entry_limit}")
    circuit = build_architecture_fused_conversion(
        config,
        explicit_primitives=True,
        decoder=decoder,
        weighted_sum_backend=weighted_sum_backend,
        reduction_backend=reduction_backend,
        reduction_cleanup=reduction_cleanup,
        lookup_entry_limit=lookup_entry_limit,
    )
    offset = config.n
    for prime, width in zip(config.moduli, config.widths, strict=True):
        field = list(circuit.qubits[offset : offset + width])
        append_block(circuit, build_mosca_zalka_inplace_qft(prime, arithmetic=local_backend), field)
        offset += width
    if coherent:
        append_block(circuit, build_general_c(config.moduli).inverse(), circuit.qubits[:offset])
    circuit.name = "sparse_mixed_coherent" if coherent else "sparse_mixed_terminal"
    circuit.metadata = {
        **(circuit.metadata or {}),
        "moduli": list(config.moduli),
        "n": config.n,
        "w": config.w,
        "coherent": coherent,
        "hamming_promise": True,
        "output_wires": list(range(config.n)) if coherent else list(range(config.n, offset)),
        "local_backend": local_backend,
        "weighted_sum_backend": weighted_sum_backend,
        "decoder": decoder,
        "lookup_entry_limit": lookup_entry_limit,
    }
    return circuit
