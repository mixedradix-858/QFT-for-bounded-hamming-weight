"""Controls circuits and mathematical helpers."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def mcx_workspace_size(num_controls: int) -> int:
    """Return the clean scratch count for the balanced multi-control X network."""
    if num_controls < 1:
        raise ValueError("an MCX needs at least one control")
    return 0 if num_controls <= 2 else num_controls - 1


def _normalized_ctrl_state(num_controls: int, ctrl_state: int | None) -> int:
    """Encode a default all-ones match or validate an explicit control pattern."""
    if ctrl_state is None:
        return (1 << num_controls) - 1
    if ctrl_state < 0 or ctrl_state >= 1 << num_controls:
        raise ValueError("ctrl_state does not fit the control register")
    return ctrl_state


def _toggle_zero_controls(
    circuit: QuantumCircuit, controls: Sequence[Qubit], ctrl_state: int
) -> None:
    """Conjugate zero-valued controls with X so the match uses positive controls."""
    for index, control in enumerate(controls):
        if ctrl_state >> index & 1 == 0:
            circuit.x(control)


def append_log_depth_mcx(
    circuit: QuantumCircuit,
    controls: Sequence[Qubit],
    target: Qubit,
    workspace: Sequence[Qubit] = (),
    ctrl_state: int | None = None,
) -> None:
    """Append an exact MCX with a balanced clean-ancilla AND tree."""
    controls = tuple(controls)
    num_controls = len(controls)
    if num_controls < 1:
        raise ValueError("an MCX needs at least one control")
    required = mcx_workspace_size(num_controls)
    if len(workspace) < required:
        raise ValueError(f"{num_controls} controls require {required} clean workspace qubits")
    state = _normalized_ctrl_state(num_controls, ctrl_state)
    _toggle_zero_controls(circuit, controls, state)
    if num_controls == 1:
        circuit.cx(controls[0], target)
    elif num_controls == 2:
        circuit.ccx(controls[0], controls[1], target)
    else:
        current = list(controls)
        operations: list[tuple[Qubit, Qubit, Qubit]] = []
        workspace_cursor = 0
        while len(current) > 1:
            next_level: list[Qubit] = []
            for cursor in range(0, len(current), 2):
                if cursor + 1 == len(current):
                    next_level.append(current[cursor])
                    continue
                parent = workspace[workspace_cursor]
                workspace_cursor += 1
                left = current[cursor]
                right = current[cursor + 1]
                circuit.ccx(left, right, parent)
                operations.append((left, right, parent))
                next_level.append(parent)
            current = next_level
        circuit.cx(current[0], target)
        for left, right, parent in reversed(operations):
            circuit.ccx(left, right, parent)
    _toggle_zero_controls(circuit, controls, state)


def append_log_depth_controlled_ry(
    circuit: QuantumCircuit,
    angle: float,
    controls: Sequence[Qubit],
    target: Qubit,
    workspace: Sequence[Qubit] = (),
    ctrl_state: int | None = None,
) -> None:
    """Apply RY(angle) conditioned by a balanced conjunction of controls."""
    controls = tuple(controls)
    if not controls:
        circuit.ry(angle, target)
        return
    state = _normalized_ctrl_state(len(controls), ctrl_state)
    required = controlled_ry_workspace_size(len(controls))
    if len(workspace) < required:
        raise ValueError(f"{len(controls)} controls require {required} workspace qubits")
    _toggle_zero_controls(circuit, controls, state)
    if len(controls) == 1:
        circuit.cry(angle, controls[0], target)
    elif len(controls) == 2:
        if not workspace:
            raise ValueError("a two-control RY requires one workspace qubit")
        circuit.ccx(controls[0], controls[1], workspace[0])
        circuit.cry(angle, workspace[0], target)
        circuit.ccx(controls[0], controls[1], workspace[0])
    else:
        current = list(controls)
        operations: list[tuple[Qubit, Qubit, Qubit]] = []
        workspace_cursor = 0
        while len(current) > 1:
            next_level: list[Qubit] = []
            for cursor in range(0, len(current), 2):
                if cursor + 1 == len(current):
                    next_level.append(current[cursor])
                    continue
                parent = workspace[workspace_cursor]
                workspace_cursor += 1
                left = current[cursor]
                right = current[cursor + 1]
                circuit.ccx(left, right, parent)
                operations.append((left, right, parent))
                next_level.append(parent)
            current = next_level
        circuit.cry(angle, current[0], target)
        for left, right, parent in reversed(operations):
            circuit.ccx(left, right, parent)
    _toggle_zero_controls(circuit, controls, state)


def controlled_ry_workspace_size(num_controls: int) -> int:
    """Return the clean conjunction scratch count for a controlled RY."""
    if num_controls <= 1:
        return 0
    if num_controls == 2:
        return 1
    return num_controls - 1
