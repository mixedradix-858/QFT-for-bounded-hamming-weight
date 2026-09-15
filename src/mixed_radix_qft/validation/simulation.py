"""Optional bounded-support evolution of explicit primitive circuits."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def simulate_classical_reversible_circuit(circuit, initial_bits: list[int]) -> list[int]:
    """Follow one basis state through an X/CX/CCX-only circuit."""
    if len(initial_bits) > circuit.num_qubits or any((bit not in (0, 1) for bit in initial_bits)):
        raise ValueError("initial_bits must fit the circuit and contain only bits")
    if float(circuit.global_phase) % (2 * np.pi):
        raise ValueError("the bit simulator cannot represent a global phase")
    state = initial_bits + [0] * (circuit.num_qubits - len(initial_bits))
    for instruction in circuit.data:
        name = instruction.operation.name
        indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if name == "x":
            state[indices[0]] ^= 1
        elif name == "cx":
            state[indices[1]] ^= state[indices[0]]
        elif name == "ccx":
            state[indices[2]] ^= state[indices[0]] & state[indices[1]]
        else:
            raise AssertionError(f"non-classical primitive {name!r}")
    return state


def simulate_sparse_state(
    circuit,
    initial: dict[int, complex],
    *,
    max_states: int = 65536,
    prune_tolerance: float = 1e-12,
    diagnostics: dict[str, float] | None = None,
) -> dict[int, complex]:
    """Evolve a sparse state through the exact gates used by arithmetic.

    Track complete physical basis indices and global phase. Report the sum
    of discarded L2 norms and stop if support exceeds max_states."""
    if max_states < 1 or not np.isfinite(prune_tolerance) or prune_tolerance < 0:
        raise ValueError("max_states must be positive and pruning finite/non-negative")
    if any((index < 0 or index >= 1 << circuit.num_qubits for index in initial)):
        raise ValueError("initial support does not fit the circuit")
    if any((not np.isfinite(value) for value in initial.values())):
        raise ValueError("initial amplitudes must be finite")
    state = {
        index: value * np.exp(1j * float(circuit.global_phase))
        for index, value in initial.items()
        if value != 0
    }
    if len(state) > max_states:
        raise MemoryError("sparse state exceeds max_states")
    discarded_bound = 0.0
    peak_states = len(state)
    for instruction in circuit.data:
        name = instruction.operation.name
        indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        next_state: defaultdict[int, complex] = defaultdict(complex)
        if name in {"x", "cx", "ccx", "swap"}:
            for basis, amplitude in state.items():
                if name == "x":
                    output = basis ^ 1 << indices[0]
                elif name == "cx":
                    output = basis ^ (basis >> indices[0] & 1) << indices[1]
                elif name == "ccx":
                    controls = basis >> indices[0] & 1 & (basis >> indices[1] & 1)
                    output = basis ^ controls << indices[2]
                else:
                    difference = basis >> indices[0] & 1 ^ basis >> indices[1] & 1
                    output = basis ^ difference << indices[0] ^ difference << indices[1]
                next_state[output] += amplitude
        elif name == "h":
            mask = 1 << indices[0]
            for basis, amplitude in state.items():
                if basis & mask:
                    next_state[basis ^ mask] += amplitude / np.sqrt(2)
                    next_state[basis] -= amplitude / np.sqrt(2)
                else:
                    next_state[basis] += amplitude / np.sqrt(2)
                    next_state[basis | mask] += amplitude / np.sqrt(2)
        elif name in {"p", "cp"}:
            angle = float(instruction.operation.params[0])
            for basis, amplitude in state.items():
                active = all((basis >> index & 1 for index in indices))
                next_state[basis] += amplitude * (np.exp(1j * angle) if active else 1)
        elif name in {"ry", "cry"}:
            angle = float(instruction.operation.params[0])
            cosine, sine = (np.cos(angle / 2), np.sin(angle / 2))
            mask = 1 << indices[-1]
            for basis, amplitude in state.items():
                if name == "cry" and (not basis >> indices[0] & 1):
                    next_state[basis] += amplitude
                else:
                    next_state[basis] += cosine * amplitude
                    next_state[basis ^ mask] += (-sine if basis & mask else sine) * amplitude
        elif name == "z":
            for basis, amplitude in state.items():
                next_state[basis] += (-1 if basis >> indices[0] & 1 else 1) * amplitude
        else:
            raise AssertionError(f"unsupported sparse gate {name!r}")
        discarded_bound += np.sqrt(
            sum((abs(value) ** 2 for value in next_state.values() if abs(value) <= prune_tolerance))
        )
        peak_states = max(peak_states, len(next_state))
        state = {
            basis: amplitude
            for basis, amplitude in next_state.items()
            if abs(amplitude) > prune_tolerance
        }
        if len(state) > max_states:
            raise MemoryError("sparse simulation exceeded max_states; no backend substitution")
    if diagnostics is not None:
        diagnostics.update(discarded_l2_bound=float(discarded_bound), peak_states=peak_states)
    return state
