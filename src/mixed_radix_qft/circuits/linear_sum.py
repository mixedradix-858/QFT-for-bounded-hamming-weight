"""Linear sum circuits and mathematical helpers."""

from __future__ import annotations

from collections import defaultdict

from mixed_radix_qft.circuits.adders.kogge_stone import (
    append_kogge_stone_sum,
    kogge_stone_prefix_workspace_size,
)
from mixed_radix_qft.circuits.adders.qfa2 import append_qfa2_word_compressor
from mixed_radix_qft.circuits.fanout import fanout
from mixed_radix_qft.circuits.workspace import fresh


def raw_weighted_sum(circuit, bits, coefficients, width):
    """Compute sum(c_i*b_i) modulo 2**width, retaining reversible history."""
    if width < 1 or len(bits) != len(coefficients):
        raise ValueError("positive width and matching coefficients required")
    words, destinations = ([], defaultdict(list))
    for bit, coefficient in zip(bits, coefficients):
        coefficient %= 1 << width
        if not coefficient:
            continue
        word = fresh(circuit, width)
        words.append(word)
        for position in range(width):
            if coefficient >> position & 1:
                destinations[bit].append(word[position])
    for source, targets in destinations.items():
        fanout(circuit, source, targets)
    if not words:
        return fresh(circuit, width)
    while len(words) > 2:
        level = []
        for index in range(0, len(words) - 2, 3):
            first, second, third = words[index : index + 3]
            carry = fresh(circuit, width)
            append_qfa2_word_compressor(circuit, first, second, third, carry)
            level.extend((third, carry))
        level.extend(words[len(words) // 3 * 3 :])
        words = level
    if len(words) == 1:
        return words[0]
    output = fresh(circuit, width)
    work = fresh(circuit, kogge_stone_prefix_workspace_size(width))
    append_kogge_stone_sum(circuit, *words, output, work)
    return output
