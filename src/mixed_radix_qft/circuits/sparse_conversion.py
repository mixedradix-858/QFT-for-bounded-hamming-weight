"""Sparse conversion circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumCircuit

from mixed_radix_qft.circuits.decoders import _append_crt_inverse_decoder_xor
from mixed_radix_qft.circuits.lookup import (
    _append_parallel_lookup_xor,
    _append_projected_decoder_xor,
    _append_serial_lookup_xor,
)
from mixed_radix_qft.circuits.residues import build_architecture_c_res
from mixed_radix_qft.classical.decoder_tables import fused_decoder_entries, projected_decoder_plan
from mixed_radix_qft.config import SparseQFTConfig


def build_architecture_fused_conversion(
    config: SparseQFTConfig,
    parallel_decoder: bool = True,
    explicit_primitives: bool = False,
    decoder: str = "lookup",
    weighted_sum_backend: str = "wallace-qfa2",
    *,
    reduction_backend: str = "cuccaro",
    reduction_cleanup: str = "local",
    lookup_entry_limit: int = 100000,
) -> QuantumCircuit:
    """Map |x>|0_Y>|0_work> to |0_X>|eta(x)>|0_work>.

    Requires x<m and Ham(x)<=w. The input decoder erases X before
    local Fourier transforms; eta_j(x)=g_j*x mod m_j."""
    projection = (
        projected_decoder_plan(config, max_entries=lookup_entry_limit)
        if decoder == "lookup-projected"
        else None
    )
    circuit = build_architecture_c_res(
        config,
        explicit_primitives=explicit_primitives,
        weighted_sum_backend=weighted_sum_backend,
        reduction_backend=reduction_backend,
        reduction_cleanup=reduction_cleanup,
    )
    x_register = circuit.qregs[0]
    y_register = circuit.qregs[1]
    if decoder in {"crt-inverse", "crt-kogge-stone", "crt-wallace-kogge-stone"}:
        _append_crt_inverse_decoder_xor(
            circuit,
            y_register,
            x_register,
            config,
            prefix="sklansky" if decoder == "crt-inverse" else "kogge-stone",
            sum_backend="wallace-qfa2" if decoder == "crt-wallace-kogge-stone" else "prefix-tree",
        )
    elif decoder == "lookup-projected":
        _append_projected_decoder_xor(
            circuit,
            y_register,
            x_register,
            projection,
            parallel=parallel_decoder,
            explicit_matches=explicit_primitives,
        )
    elif decoder == "lookup" and parallel_decoder:
        _append_parallel_lookup_xor(
            circuit,
            y_register,
            x_register,
            fused_decoder_entries(config),
            "dec",
            explicit_matches=explicit_primitives,
        )
    elif decoder == "lookup":
        _append_serial_lookup_xor(
            circuit,
            y_register,
            x_register,
            fused_decoder_entries(config),
            explicit_matches=explicit_primitives,
        )
    else:
        raise ValueError(
            "decoder must be lookup, lookup-projected, crt-inverse, crt-kogge-stone, "
            "or crt-wallace-kogge-stone"
        )
    circuit.name = f"C_sparse_A_architecture_{decoder}_{weighted_sum_backend}"
    circuit.metadata = {
        **(circuit.metadata or {}),
        "decoder": decoder,
        "reduction_backend": reduction_backend,
        "reduction_cleanup": reduction_cleanup,
    }
    return circuit
