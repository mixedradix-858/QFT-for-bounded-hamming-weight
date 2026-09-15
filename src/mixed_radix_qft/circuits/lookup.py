"""Lookup circuits and mathematical helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.controls import append_log_depth_mcx, mcx_workspace_size
from mixed_radix_qft.circuits.fanout import _fanout_to_targets, _uncompute_fanout


def _append_serial_lookup_xor(
    circuit: QuantumCircuit,
    source_qubits: Sequence[Qubit],
    target_qubits: Sequence[Qubit],
    entries: Iterable[tuple[int, int]],
    explicit_matches: bool = False,
) -> None:
    """Append target ^= table[source] as a serial constant lookup."""
    controls = list(source_qubits)
    workspace = ()
    if explicit_matches and mcx_workspace_size(len(controls)):
        work = QuantumRegister(mcx_workspace_size(len(controls)), "serial_match_work")
        circuit.add_register(work)
        workspace = tuple(work)
    for source_value, target_value in entries:
        for bit, target in enumerate(target_qubits):
            if target_value >> bit & 1:
                if explicit_matches:
                    append_log_depth_mcx(
                        circuit, controls, target, workspace, ctrl_state=source_value
                    )
                else:
                    circuit.mcx(controls, target, ctrl_state=source_value)


def _append_projected_decoder_xor(
    circuit, source_qubits, target_qubits, plan, *, parallel=True, explicit_matches=True
):
    """Append X <- X XOR d_J(P_J(Y)) with the existing clean lookup networks."""
    if len(source_qubits) != plan.source_width or any(
        (value >= 1 << len(target_qubits) for _, value in plan.entries)
    ):
        raise ValueError("projection source width or XOR target width mismatch")
    sources = [source_qubits[bit] for bit in plan.positions]
    if not sources:
        for _, value in plan.entries:
            for bit, target in enumerate(target_qubits):
                if value >> bit & 1:
                    circuit.x(target)
    elif parallel:
        _append_parallel_lookup_xor(
            circuit,
            sources,
            target_qubits,
            plan.entries,
            "dec_projected",
            explicit_matches=explicit_matches,
        )
    else:
        _append_serial_lookup_xor(
            circuit, sources, target_qubits, plan.entries, explicit_matches=explicit_matches
        )
    circuit.metadata = {
        **(circuit.metadata or {}),
        "decoder": "lookup-projected",
        "lookup_source_width": plan.source_width,
        "lookup_key_width": len(plan.positions),
        "lookup_entries": len(plan.entries),
        "lookup_projection_bits": list(plan.positions),
        "decoder_domain": "promised input only; off-domain extension may differ",
    }


def _xor_tree_into_target(
    circuit: QuantumCircuit, controls: Sequence[Qubit], target: Qubit, label: str
) -> None:
    """XOR controls into one target with a balanced tree, then clean the tree."""
    if not controls:
        return
    if len(controls) == 1:
        circuit.cx(controls[0], target)
        return
    current = list(controls)
    levels: list[list[tuple[Qubit, Qubit, Qubit]]] = []
    level = 0
    while len(current) > 1:
        pairs = len(current) // 2
        parents = QuantumRegister(pairs, f"{label}_xor_l{level}")
        circuit.add_register(parents)
        next_level: list[Qubit] = []
        level_operations: list[tuple[Qubit, Qubit, Qubit]] = []
        parent_index = 0
        for cursor in range(0, len(current), 2):
            if cursor + 1 == len(current):
                next_level.append(current[cursor])
                continue
            left = current[cursor]
            right = current[cursor + 1]
            parent = parents[parent_index]
            circuit.cx(left, parent)
            circuit.cx(right, parent)
            next_level.append(parent)
            level_operations.append((left, right, parent))
            parent_index += 1
        levels.append(level_operations)
        current = next_level
        level += 1
    circuit.cx(current[0], target)
    for operations in reversed(levels):
        for left, right, parent in reversed(operations):
            circuit.cx(right, parent)
            circuit.cx(left, parent)


def _append_parallel_lookup_xor(
    circuit: QuantumCircuit,
    source_qubits: Sequence[Qubit],
    target_qubits: Sequence[Qubit],
    entries: Iterable[tuple[int, int]],
    label: str,
    explicit_matches: bool = False,
) -> None:
    """Append |s>|t>|0_work> -> |s>|t XOR table[s]>|0_work>.

    Unlisted keys map to zero. Parallel equality flags feed balanced XOR trees."""
    active_entries = tuple(
        ((source_value, target_value) for source_value, target_value in entries if target_value)
    )
    if not active_entries:
        return
    source_width = len(source_qubits)
    target_width = len(target_qubits)
    entry_count = len(active_entries)
    source_copies = QuantumRegister(entry_count * source_width, f"{label}_source")
    flags = QuantumRegister(entry_count, f"{label}_flags")
    circuit.add_register(source_copies)
    circuit.add_register(flags)

    def source_copy(entry: int, bit: int) -> Qubit:
        """Locate a bit in the private source copy for one parallel lookup entry."""
        return source_copies[entry * source_width + bit]

    source_fanouts: list[list[tuple[Qubit, Qubit]]] = []
    for bit, source in enumerate(source_qubits):
        targets = [source_copy(entry, bit) for entry in range(entry_count)]
        source_fanouts.append(_fanout_to_targets(circuit, source, targets))
    match_workspace_width = mcx_workspace_size(source_width) if explicit_matches else 0
    match_workspace: QuantumRegister | None = None
    if match_workspace_width:
        match_workspace = QuantumRegister(
            entry_count * match_workspace_width, f"{label}_match_work"
        )
        circuit.add_register(match_workspace)

    def entry_match_workspace(entry: int) -> list[Qubit]:
        """Return the clean conjunction scratch slice for one lookup entry."""
        if match_workspace is None:
            return []
        start = entry * match_workspace_width
        return [match_workspace[start + offset] for offset in range(match_workspace_width)]

    for entry, (source_value, _) in enumerate(active_entries):
        controls = [source_copy(entry, bit) for bit in range(source_width)]
        if explicit_matches:
            append_log_depth_mcx(
                circuit,
                controls,
                flags[entry],
                entry_match_workspace(entry),
                ctrl_state=source_value,
            )
        else:
            circuit.mcx(controls, flags[entry], ctrl_state=source_value)
    copy_specs = [
        (entry, bit)
        for entry, (_, target_value) in enumerate(active_entries)
        for bit in range(target_width)
        if target_value >> bit & 1
    ]
    flag_copies = QuantumRegister(len(copy_specs), f"{label}_flag_copies")
    circuit.add_register(flag_copies)
    by_entry: dict[int, list[Qubit]] = {entry: [] for entry in range(entry_count)}
    by_target: dict[int, list[Qubit]] = {bit: [] for bit in range(target_width)}
    for copy_index, (entry, bit) in enumerate(copy_specs):
        by_entry[entry].append(flag_copies[copy_index])
        by_target[bit].append(flag_copies[copy_index])
    flag_fanouts = [
        _fanout_to_targets(circuit, flags[entry], by_entry[entry]) for entry in range(entry_count)
    ]
    for bit, target in enumerate(target_qubits):
        _xor_tree_into_target(circuit, by_target[bit], target, f"{label}_t{bit}")
    for operations in reversed(flag_fanouts):
        _uncompute_fanout(circuit, operations)
    for entry, (source_value, _) in reversed(tuple(enumerate(active_entries))):
        controls = [source_copy(entry, bit) for bit in range(source_width)]
        if explicit_matches:
            append_log_depth_mcx(
                circuit,
                controls,
                flags[entry],
                entry_match_workspace(entry),
                ctrl_state=source_value,
            )
        else:
            circuit.mcx(controls, flags[entry], ctrl_state=source_value)
    for operations in reversed(source_fanouts):
        _uncompute_fanout(circuit, operations)
