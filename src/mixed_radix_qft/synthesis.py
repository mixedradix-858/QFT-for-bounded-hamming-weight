"""Select explicit constructions separately from Qiskit basis transpilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from qiskit import QuantumCircuit, transpile

from mixed_radix_qft.algorithms.general import build_general_mixed_qft
from mixed_radix_qft.algorithms.sparse import build_sparse_qft
from mixed_radix_qft.circuits.local_qft_components.modes import EXPLICIT_ARITHMETIC_MODES
from mixed_radix_qft.classical.number_theory import validate_moduli
from mixed_radix_qft.config import SparseQFTConfig


@dataclass(frozen=True)
class SynthesisOptions:
    """Explicit local and sparse arithmetic choices; None means not applicable."""

    local_backend: str = "explicit-register-sklansky"
    weighted_sum_backend: str | None = None
    decoder: str | None = None
    reduction_backend: str | None = None
    reduction_cleanup: str | None = None
    lookup_entry_limit: int | None = None


def build_qft(
    algorithm: str,
    moduli: Sequence[int],
    *,
    w: int | None = None,
    coherent: bool = True,
    synthesis: SynthesisOptions | None = None,
) -> QuantumCircuit:
    """Build F_m on valid inputs, or its terminal CRT-encoded output.

    General inputs satisfy x<m; sparse inputs also satisfy Ham(x)<=w.
    Public inputs start on X, and every other register starts in zero.
    """
    options = synthesis or SynthesisOptions()
    factors = tuple(moduli)
    validate_moduli(factors)
    if options.local_backend not in EXPLICIT_ARITHMETIC_MODES:
        raise ValueError("unsupported explicit local synthesis")
    if algorithm == "general":
        if w is not None or any(
            value is not None for key, value in asdict(options).items() if key != "local_backend"
        ):
            raise ValueError(
                "w, weighted-sum, decoder, reduction and lookup options apply only to sparse QFT"
            )
        circuit = build_general_mixed_qft(
            factors, coherent=coherent, local_backend=options.local_backend
        )
        effective = {
            "local_backend": options.local_backend,
            "general_arithmetic": "wallace-qfa2-kogge-stone",
        }
    elif algorithm == "sparse":
        if w is None:
            raise ValueError("sparse QFT requires a Hamming-weight bound w")
        effective = {
            "local_backend": options.local_backend,
            "weighted_sum_backend": options.weighted_sum_backend
            if options.weighted_sum_backend is not None
            else "wallace-qfa2",
            "decoder": options.decoder if options.decoder is not None else "lookup",
            "reduction_backend": options.reduction_backend
            if options.reduction_backend is not None
            else "cuccaro",
            "reduction_cleanup": options.reduction_cleanup
            if options.reduction_cleanup is not None
            else "local",
            "lookup_entry_limit": options.lookup_entry_limit
            if options.lookup_entry_limit is not None
            else 100_000,
        }
        circuit = build_sparse_qft(
            SparseQFTConfig.from_moduli(factors, w), coherent=coherent, **effective
        )
    else:
        raise ValueError("algorithm must be general or sparse")
    circuit.metadata = {**(circuit.metadata or {}), "algorithm": algorithm, "synthesis": effective}
    return circuit


def transpile_circuit(
    circuit: QuantumCircuit, *, basis: str = "u-cx", optimization_level: int = 1, seed: int = 7
) -> QuantumCircuit:
    """Lower the constructed unitary to a specified one/two-qubit basis."""
    if basis not in {"u-cx", "rz-sx-x-cx"}:
        raise ValueError("basis must be u-cx or rz-sx-x-cx")
    if optimization_level not in range(4):
        raise ValueError("optimization_level must be 0, 1, 2 or 3")
    gates = ["u", "cx"] if basis == "u-cx" else ["rz", "sx", "x", "cx"]
    result = transpile(
        circuit, basis_gates=gates, optimization_level=optimization_level, seed_transpiler=seed
    )
    result.metadata = {
        **(result.metadata or {}),
        "transpilation": {
            "basis": gates,
            "optimization_level": optimization_level,
            "seed": seed,
            "topology": "all-to-all",
        },
    }
    return result
