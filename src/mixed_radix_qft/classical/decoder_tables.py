"""Decoder tables circuits and mathematical helpers."""

from __future__ import annotations

from numbers import Integral

from mixed_radix_qft.classical.crt import encode_fields, transformed_crt_tuple
from mixed_radix_qft.classical.projected_lookup import LookupProjection, project_lookup_table
from mixed_radix_qft.classical.sparse_domain import sparse_value_count
from mixed_radix_qft.config import SparseQFTConfig


def fused_residue_entries(config: SparseQFTConfig) -> tuple[tuple[int, int], ...]:
    """Enumerate the lookup pairs x -> packed eta(x) on the promised domain."""
    return tuple(
        (
            (value, encode_fields(transformed_crt_tuple(value, config.moduli), config.widths))
            for value in config.sparse_domain
        )
    )


def fused_decoder_entries(config: SparseQFTConfig) -> tuple[tuple[int, int], ...]:
    """Reverse the promised lookup pairs to decode eta(x) into x."""
    return tuple(((label, value) for value, label in fused_residue_entries(config)))


def projected_decoder_plan(
    config: SparseQFTConfig, *, max_entries: int = 100000
) -> LookupProjection:
    """Select and certify a shorter eta key for ANY supported Hamming bound."""
    config.validate()
    if not isinstance(max_entries, Integral) or isinstance(max_entries, bool) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    count = sparse_value_count(config.n, config.w, config.crt_modulus)
    if count > max_entries:
        raise MemoryError(
            f"projected lookup has {count} promised entries, limit {max_entries}; no enumeration, circuit construction or backend fallback performed"
        )
    return project_lookup_table(fused_decoder_entries(config), config.q_y)
