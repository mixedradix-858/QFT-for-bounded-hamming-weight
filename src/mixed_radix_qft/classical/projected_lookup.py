"""Projected lookup circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class LookupProjection:
    """Certified distinct projected keys; positions use original little-endian indices."""

    source_width: int
    positions: tuple[int, ...]
    entries: tuple[tuple[int, int], ...]


def project_key(key, positions):
    """Pack selected input bits in the order supplied, little-endian first."""
    positions = tuple(positions)
    if (
        not isinstance(key, Integral)
        or isinstance(key, bool)
        or key < 0
        or any((not isinstance(p, Integral) or isinstance(p, bool) or p < 0 for p in positions))
        or (len(set(positions)) != len(positions))
    ):
        raise ValueError("nonnegative integer key and distinct nonnegative bit positions required")
    return sum(((int(key) >> int(bit) & 1) << index for index, bit in enumerate(positions)))


def select_injective_bits(keys, source_width):
    """Greedily separate every distinct key, then certify exact injectivity."""
    if not isinstance(source_width, Integral) or isinstance(source_width, bool) or source_width < 0:
        raise ValueError("source_width must be a nonnegative integer")
    source_width = int(source_width)
    keys = tuple(keys)
    if any(
        (
            not isinstance(key, Integral)
            or isinstance(key, bool)
            or key < 0
            or (key >= 1 << source_width)
            for key in keys
        )
    ):
        raise ValueError("keys must be nonnegative integers fitting source_width")
    keys = tuple((int(key) for key in keys))
    if len(set(keys)) != len(keys):
        raise ValueError("projection requires distinct original keys")
    groups = [keys] if len(keys) > 1 else []
    remaining, selected = (list(range(source_width)), [])
    while groups:
        best_bit, best_score = (None, 0)
        for bit in remaining:
            score = 0
            for group in groups:
                ones = sum((key >> bit & 1 for key in group))
                score += ones * (len(group) - ones)
            if score > best_score:
                best_bit, best_score = (bit, score)
        if best_bit is None:
            raise RuntimeError("distinct full keys could not be separated")
        selected.append(best_bit)
        remaining.remove(best_bit)
        divided = []
        for group in groups:
            zero = tuple((key for key in group if not key >> best_bit & 1))
            one = tuple((key for key in group if key >> best_bit & 1))
            divided.extend((part for part in (zero, one) if len(part) > 1))
        groups = divided
    positions = tuple(sorted(selected))
    if len({project_key(key, positions) for key in keys}) != len(keys):
        raise RuntimeError("projection failed its final injectivity certificate")
    return positions


def project_lookup_table(entries, source_width):
    """Certify and project an entire finite XOR table without dropping zero rows."""
    entries = tuple(entries)
    if any(
        (
            not isinstance(value, Integral) or isinstance(value, bool) or value < 0
            for _, value in entries
        )
    ):
        raise ValueError("lookup outputs must be nonnegative integers")
    positions = select_injective_bits((key for key, _ in entries), source_width)
    return LookupProjection(
        int(source_width),
        positions,
        tuple(((project_key(key, positions), int(value)) for key, value in entries)),
    )
