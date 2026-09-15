"""Bounded reduction circuits and mathematical helpers."""

from __future__ import annotations

from numbers import Integral

from mixed_radix_qft.circuits.adders.cuccaro import append_controlled_cuccaro_add
from mixed_radix_qft.circuits.adders.kogge_stone import (
    append_kogge_stone_sum,
    kogge_stone_prefix_workspace_size,
)
from mixed_radix_qft.circuits.adders.qfa2 import append_qfa2_word_compressor
from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.circuits.fanout import fanout
from mixed_radix_qft.circuits.workspace import fresh, undo

REDUCTION_BACKENDS = ("cuccaro", "prefix")


REDUCTION_CLEANUP_MODES = ("local", "deferred")


def validate_reduction_options(backend, cleanup, explicit_primitives=True):
    """Reject unknown variants or a new variant in a non-explicit circuit."""
    if backend not in REDUCTION_BACKENDS or cleanup not in REDUCTION_CLEANUP_MODES:
        raise ValueError("reduction requires cuccaro/prefix and local/deferred cleanup")
    if not explicit_primitives and (backend, cleanup) != ("cuccaro", "local"):
        raise ValueError(
            "new reduction variants require explicit_primitives=True / architecture-explicit"
        )


def _threshold_flags(circuit, source, modulus, count, parallel):
    """Return flags [S>=t*p], retaining any private copies of S for inversion."""
    width = len(source)
    flags = fresh(circuit, count)
    sources = [source] * count
    if parallel and count > 1:
        sources = [fresh(circuit, width) for _ in flags]
        for bit in range(width):
            fanout(circuit, source[bit], [word[bit] for word in sources])
    for t, (word, flag) in enumerate(zip(sources, flags), 1):
        append_log_depth_constant_comparator(
            circuit, word, flag, t * modulus, fresh(circuit, comparator_workspace_size(width))
        )
    return flags


def _prefix_correction(circuit, source, flags, modulus):
    """Compute (S-p*sum(flags)) mod 2**b with carry-save and one prefix sum."""
    width = len(source)
    words = [source]
    if len(flags) > 1:
        words = [fresh(circuit, width)]
        for left, right in zip(source, words[0]):
            circuit.cx(left, right)
    constant = -modulus % (1 << width)
    for flag in flags:
        word = fresh(circuit, width)
        fanout(circuit, flag, [word[i] for i in range(width) if constant >> i & 1])
        words.append(word)
    while len(words) > 2:
        level = []
        for index in range(0, len(words) - 2, 3):
            first, second, third = words[index : index + 3]
            carry = fresh(circuit, width)
            append_qfa2_word_compressor(circuit, first, second, third, carry)
            level.extend((third, carry))
        level.extend(words[len(words) // 3 * 3 :])
        words = level
    result = fresh(circuit, width)
    append_kogge_stone_sum(
        circuit, *words, result, fresh(circuit, kogge_stone_prefix_workspace_size(width))
    )
    return result


def append_bounded_reduction(circuit, source_sum, modulus, w, *, backend="prefix", cleanup="local"):
    """Compute S mod p, preserving 0<=S<=w*(p-1), with w>=1.

    Return residue wires. Local cleanup leaves only this result nonzero;
    deferred cleanup retains history, and result wires may alias S.
    Copy the result before reversing the entire deferred stage.
    """
    validate_reduction_options(backend, cleanup)
    source = list(source_sum)
    width = len(source)
    if (
        not isinstance(modulus, Integral)
        or isinstance(modulus, bool)
        or modulus < 2
        or (not isinstance(w, Integral))
        or isinstance(w, bool)
        or (w < 1)
    ):
        raise ValueError("require integer modulus>=2 and w>=1")
    if not width or len(set(source)) != width or w * (modulus - 1) >= 1 << width:
        raise ValueError("distinct source wires wide enough for the promised sum are required")
    residue_width = int(modulus - 1).bit_length()
    count = w * (modulus - 1) // modulus
    start = len(circuit.data)
    result = source
    if count:
        if backend == "cuccaro":
            result = fresh(circuit, width)
            for left, right in zip(source, result):
                circuit.cx(left, right)
        flags = _threshold_flags(circuit, source, modulus, count, backend == "prefix")
        if backend == "prefix":
            result = _prefix_correction(circuit, source, flags, modulus)
        else:
            constant = fresh(circuit, width)
            for bit in range(width):
                if -modulus % (1 << width) >> bit & 1:
                    circuit.x(constant[bit])
            helper, mcx_work = (fresh(circuit, 1), fresh(circuit, 2))
            for flag in flags:
                append_controlled_cuccaro_add(circuit, flag, constant, result, helper[0], mcx_work)
    stop = len(circuit.data)
    if cleanup == "local":
        output = fresh(circuit, residue_width)
        for left, right in zip(result, output):
            circuit.cx(left, right)
        undo(circuit, start, stop)
        return tuple(output)
    return tuple(result[:residue_width])
