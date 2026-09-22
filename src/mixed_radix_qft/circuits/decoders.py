"""Decoders circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.adders.kogge_stone import (
    append_kogge_stone_difference,
    append_kogge_stone_sum,
    kogge_stone_prefix_workspace_size,
)
from mixed_radix_qft.circuits.adders.sklansky import (
    append_sklansky_difference,
    append_sklansky_sum,
    sklansky_prefix_workspace_size,
)
from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.circuits.fanout import _append_controlled_binary_constant, _fanout_to_targets
from mixed_radix_qft.circuits.linear_sum import raw_weighted_sum
from mixed_radix_qft.config import SparseQFTConfig


def _append_prefix_sum_tree(
    circuit: QuantumCircuit, sources: Sequence[Sequence[Qubit]], label: str, prefix: str
) -> tuple[Qubit, ...]:
    """Return the root of an exact-width balanced prefix-adder tree."""
    if not sources:
        raise ValueError("at least one source register is required")
    current = [tuple(source) for source in sources]
    width = len(current[0])
    if width <= 0 or any((len(source) != width for source in current)):
        raise ValueError("all source registers must have equal positive width")
    if prefix == "sklansky":
        workspace_width = sklansky_prefix_workspace_size(width)
        append_sum = append_sklansky_sum
    elif prefix == "kogge-stone":
        workspace_width = kogge_stone_prefix_workspace_size(width)
        append_sum = append_kogge_stone_sum
    else:
        raise ValueError("prefix must be sklansky or kogge-stone")
    level = 0
    while len(current) > 1:
        pair_count = len(current) // 2
        parents = QuantumRegister(pair_count * width, f"{label}_l{level}_sum")
        work = QuantumRegister(pair_count * workspace_width, f"{label}_l{level}_work")
        circuit.add_register(parents)
        circuit.add_register(work)
        next_level: list[tuple[Qubit, ...]] = []
        for pair in range(pair_count):
            left = current[2 * pair]
            right = current[2 * pair + 1]
            parent = tuple((parents[pair * width + bit] for bit in range(width)))
            workspace = tuple(
                (work[pair * workspace_width + bit] for bit in range(workspace_width))
            )
            append_sum(circuit, left, right, parent, workspace)
            next_level.append(parent)
        if len(current) % 2:
            next_level.append(current[-1])
        current = next_level
        level += 1
    return current[0]


@dataclass(frozen=True)
class CRTInverseDecoderBlock:
    """Arithmetic decoder workspace before outer copy-uncompute."""

    circuit: QuantumCircuit
    remainder_indices: tuple[int, ...]
    sum_width: int
    quotient_bound: int
    partial_count: int
    prefix: str
    sum_backend: str = "prefix-tree"


def build_crt_inverse_decoder_block(
    config: SparseQFTConfig, prefix: str = "sklansky", *, sum_backend: str = "prefix-tree"
) -> CRTInverseDecoderBlock:
    """Compute r=sum_j (m/m_j)*y_j mod m for valid eta fields y_j<m_j.

    Preserve Y and retain intermediate work for outer copy-uncompute.
    This eta decoder is not the ordinary CRT inverse used after the QFT.
    With wallace-qfa2, both multioperand sums use reversible carry-save
    compression and one final Kogge-Stone addition each. The entire decoder
    then has O(log n) logical depth, with sufficient all-to-all workspace.
    """
    config.validate()
    if prefix not in {"sklansky", "kogge-stone"}:
        raise ValueError("prefix must be sklansky or kogge-stone")
    if sum_backend not in {"prefix-tree", "wallace-qfa2"}:
        raise ValueError("sum_backend must be prefix-tree or wallace-qfa2")
    if sum_backend == "wallace-qfa2" and prefix != "kogge-stone":
        raise ValueError("wallace-qfa2 decoder requires the kogge-stone final adder")
    y_register = QuantumRegister(config.q_y, "y")
    circuit = QuantumCircuit(y_register, name="Dec_CRT_inverse")
    modulus_product = config.crt_modulus
    terms: list[tuple[Qubit, int]] = []
    maximum_sum = 0
    offset = 0
    for modulus, width in zip(config.moduli, config.widths, strict=True):
        factor = modulus_product // modulus
        maximum_sum += (modulus - 1) * factor
        for bit in range(width):
            terms.append((y_register[offset + bit], factor << bit))
        offset += width
    sum_width = max(1, maximum_sum.bit_length())
    if sum_backend == "wallace-qfa2":
        exact_sum = raw_weighted_sum(
            circuit, [control for control, _ in terms], [value for _, value in terms], sum_width
        )
    else:
        partials = QuantumRegister(len(terms) * sum_width, "crt_partial")
        circuit.add_register(partials)
        partial_registers: list[tuple[Qubit, ...]] = []
        for term_index, (control, value) in enumerate(terms):
            partial = tuple((partials[term_index * sum_width + bit] for bit in range(sum_width)))
            _append_controlled_binary_constant(circuit, control, partial, value)
            partial_registers.append(partial)
        exact_sum = _append_prefix_sum_tree(circuit, partial_registers, "crt_weighted", prefix)
    quotient_bound = maximum_sum // modulus_product
    quotient_multiple: tuple[Qubit, ...] | None = None
    if quotient_bound:
        sum_copies = QuantumRegister(quotient_bound * sum_width, "crt_sum_copies")
        flags = QuantumRegister(quotient_bound, "crt_threshold_flags")
        comparator_width = comparator_workspace_size(sum_width)
        comparator_work = QuantumRegister(quotient_bound * comparator_width, "crt_comparator_work")
        circuit.add_register(sum_copies)
        circuit.add_register(flags)
        circuit.add_register(comparator_work)
        copied_sums: list[tuple[Qubit, ...]] = []
        for threshold_index in range(quotient_bound):
            copied_sums.append(
                tuple((sum_copies[threshold_index * sum_width + bit] for bit in range(sum_width)))
            )
        for bit, source in enumerate(exact_sum):
            _fanout_to_targets(
                circuit,
                source,
                [copied_sums[threshold_index][bit] for threshold_index in range(quotient_bound)],
            )
        for threshold_index, copied_sum in enumerate(copied_sums):
            workspace = tuple(
                (
                    comparator_work[threshold_index * comparator_width + bit]
                    for bit in range(comparator_width)
                )
            )
            append_log_depth_constant_comparator(
                circuit,
                copied_sum,
                flags[threshold_index],
                (threshold_index + 1) * modulus_product,
                workspace,
            )
        if sum_backend == "wallace-qfa2":
            quotient_multiple = tuple(
                raw_weighted_sum(
                    circuit, list(flags), [modulus_product] * quotient_bound, sum_width
                )
            )
        else:
            multiples = QuantumRegister(quotient_bound * sum_width, "crt_modulus_multiples")
            circuit.add_register(multiples)
            multiple_registers: list[tuple[Qubit, ...]] = []
            for threshold_index, flag in enumerate(flags):
                multiple = tuple(
                    (multiples[threshold_index * sum_width + bit] for bit in range(sum_width))
                )
                _append_controlled_binary_constant(circuit, flag, multiple, modulus_product)
                multiple_registers.append(multiple)
            quotient_multiple = _append_prefix_sum_tree(
                circuit, multiple_registers, "crt_quotient", prefix
            )
    remainder = QuantumRegister(sum_width, "crt_remainder")
    circuit.add_register(remainder)
    if quotient_multiple is None:
        for source, target in zip(exact_sum, remainder, strict=True):
            circuit.cx(source, target)
    else:
        if prefix == "sklansky":
            subtract_workspace_width = sklansky_prefix_workspace_size(sum_width)
            append_difference = append_sklansky_difference
        else:
            subtract_workspace_width = kogge_stone_prefix_workspace_size(sum_width)
            append_difference = append_kogge_stone_difference
        subtract_work = QuantumRegister(subtract_workspace_width, "crt_subtract_work")
        circuit.add_register(subtract_work)
        append_difference(circuit, exact_sum, quotient_multiple, remainder, subtract_work)
    remainder_indices = tuple((circuit.find_bit(qubit).index for qubit in remainder))
    return CRTInverseDecoderBlock(
        circuit=circuit,
        remainder_indices=remainder_indices,
        sum_width=sum_width,
        quotient_bound=quotient_bound,
        partial_count=len(terms),
        prefix=prefix,
        sum_backend=sum_backend,
    )


def _append_crt_inverse_decoder_xor(
    circuit: QuantumCircuit,
    source_y: Sequence[Qubit],
    target_x: Sequence[Qubit],
    config: SparseQFTConfig,
    prefix: str,
    *,
    sum_backend: str = "prefix-tree",
) -> int:
    """Append arithmetic inverse CRT, XOR it into X, and clean all work."""
    block = build_crt_inverse_decoder_block(config, prefix=prefix, sum_backend=sum_backend)
    work_width = block.circuit.num_qubits - len(source_y)
    work = QuantumRegister(work_width, "dec_crt_work")
    circuit.add_register(work)
    mapping = [*source_y, *work]
    circuit.compose(block.circuit, qubits=mapping, inplace=True)
    remainder = [mapping[index] for index in block.remainder_indices]
    for source, target in zip(remainder, target_x):
        circuit.cx(source, target)
    circuit.compose(block.circuit.inverse(), qubits=mapping, inplace=True)
    return work_width
