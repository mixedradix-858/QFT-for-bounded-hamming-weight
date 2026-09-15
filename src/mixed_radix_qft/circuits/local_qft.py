"""Local qft circuits and mathematical helpers."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from mixed_radix_qft.circuits.local_qft_components.modes import EXPLICIT_ARITHMETIC_MODES
from mixed_radix_qft.circuits.local_qft_components.qfp import build_qfp
from mixed_radix_qft.circuits.local_qft_components.qfs import build_qfs
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime


def build_mosca_zalka_qft(
    prime: int, arithmetic: str = "explicit-register-sklansky"
) -> QuantumCircuit:
    """Map |x>|0>|0_work> to |0>|Psi_x>|0_work> by QFP† QFS.

    Requires an odd prime p and x<p; Psi_x(y)=exp(2*pi*i*x*y/p)/sqrt(p)."""
    if not is_prime(prime) or prime == 2:
        raise ValueError("use an odd prime; F_2 is a Hadamard gate")
    width = ceil_log2(prime)
    source = QuantumRegister(width, "source")
    fourier = QuantumRegister(width, "fourier")
    circuit = QuantumCircuit(source, fourier, name=f"MoscaZalkaQFT_{prime}")
    qfs = build_qfs(prime)
    qfs_work_width = qfs.num_qubits - 2 * width
    if qfs_work_width:
        qfs_work = QuantumRegister(qfs_work_width, "qfs_work")
        circuit.add_register(qfs_work)
    else:
        qfs_work = ()
    circuit.compose(qfs, qubits=[*source, *fourier, *qfs_work], inplace=True)
    qfp, layout = build_qfp(prime, arithmetic=arithmetic)
    qfp_work = QuantumRegister(len(layout.workspace_indices), "qfp_work")
    circuit.add_register(qfp_work)
    qfp_mapping: list[Qubit | None] = [None] * qfp.num_qubits
    for local_index, global_qubit in zip(layout.eigen_indices, fourier, strict=True):
        qfp_mapping[local_index] = global_qubit
    for local_index, global_qubit in zip(layout.workspace_indices, qfp_work, strict=True):
        qfp_mapping[local_index] = global_qubit
    for local_index, global_qubit in zip(layout.save_indices, source, strict=True):
        qfp_mapping[local_index] = global_qubit
    if any((qubit is None for qubit in qfp_mapping)):
        raise AssertionError("incomplete QFP qubit mapping")
    circuit.compose(
        qfp.inverse(), qubits=[qubit for qubit in qfp_mapping if qubit is not None], inplace=True
    )
    return circuit


def build_mosca_zalka_inplace_qft(
    prime: int, arithmetic: str = "explicit-register-sklansky"
) -> QuantumCircuit:
    """Map |x>|0_work> to F_p|x>|0_work> for 0<=x<p.

    Uses positive phase and little-endian fields; p must be prime, with F_2=H.
    Exact rotations are represented in floating-point arithmetic."""
    if arithmetic not in {*EXPLICIT_ARITHMETIC_MODES}:
        raise ValueError("unsupported Mosca--Zalka arithmetic backend")
    if prime == 2:
        source = QuantumRegister(1, "source")
        circuit = QuantumCircuit(source, name="MoscaZalkaQFT_2_inplace")
        circuit.h(source[0])
        return circuit
    circuit = build_mosca_zalka_qft(prime, arithmetic=arithmetic)
    width = ceil_log2(prime)
    for source_qubit, fourier_qubit in zip(circuit.qregs[0], circuit.qregs[1], strict=True):
        circuit.swap(source_qubit, fourier_qubit)
    circuit.name = f"MoscaZalkaQFT_{prime}_inplace"
    if len(circuit.qregs[0]) != width:
        raise AssertionError("unexpected source-register width")
    return circuit
