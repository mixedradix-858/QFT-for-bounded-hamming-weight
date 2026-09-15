"""Portable OpenQASM 3 serialization of an already constructed circuit."""

from pathlib import Path

from qiskit import QuantumCircuit, qasm3


def export_qasm(circuit: QuantumCircuit, path: str | Path) -> None:
    """Serialize the complete circuit, retaining all physical workspace wires."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = qasm3.dumps(circuit)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(source)
