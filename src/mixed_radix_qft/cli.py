"""Command-line circuit generation, explicit synthesis, export and small checks."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from math import prod
from pathlib import Path
from platform import python_version

from mixed_radix_qft.circuits.local_qft_components.modes import EXPLICIT_ARITHMETIC_MODES
from mixed_radix_qft.export import export_qasm
from mixed_radix_qft.synthesis import SynthesisOptions, build_qft, transpile_circuit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("algorithm", choices=("general", "sparse"))
    result.add_argument(
        "--moduli",
        type=int,
        nargs="+",
        required=True,
        help="distinct prime factors, in register order",
    )
    result.add_argument("--w", type=int, help="sparse input Hamming-weight bound")
    result.add_argument("--mode", choices=("coherent", "terminal"), default="coherent")
    result.add_argument(
        "--local-synthesis",
        choices=sorted(EXPLICIT_ARITHMETIC_MODES),
        default="explicit-register-sklansky",
    )
    result.add_argument("--weighted-sum", choices=("wallace-qfa2", "cuccaro"))
    result.add_argument(
        "--decoder", choices=("lookup", "lookup-projected", "crt-inverse", "crt-kogge-stone")
    )
    result.add_argument("--reduction", choices=("cuccaro", "prefix"))
    result.add_argument("--cleanup", choices=("local", "deferred"))
    result.add_argument("--lookup-entry-limit", type=int)
    result.add_argument("--basis", choices=("native", "u-cx", "rz-sx-x-cx"), default="native")
    result.add_argument("--optimization-level", choices=range(4), type=int)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument(
        "--output", type=Path, help="new output directory for circuit.qasm and metadata.json"
    )
    result.add_argument("--draw", action="store_true", help="print the generated circuit as text")
    result.add_argument(
        "--simulate",
        action="store_true",
        help="check a seeded complex state against the DFT (m<=32)",
    )
    result.add_argument(
        "--max-states",
        type=int,
        default=65536,
        help="maximum support for optional sparse simulation",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    if args.output and args.output.exists():
        cli.error("output directory already exists; choose a new path")
    if args.optimization_level is not None and args.basis == "native":
        cli.error("--optimization-level requires a transpiled --basis")
    if args.simulate and prod(args.moduli) > 32:
        cli.error("--simulate requires m<=32")
    if args.max_states < 1:
        cli.error("--max-states must be positive")
    try:
        options = SynthesisOptions(
            local_backend=args.local_synthesis,
            weighted_sum_backend=args.weighted_sum,
            decoder=args.decoder,
            reduction_backend=args.reduction,
            reduction_cleanup=args.cleanup,
            lookup_entry_limit=args.lookup_entry_limit,
        )
        circuit = build_qft(
            args.algorithm,
            args.moduli,
            w=args.w,
            coherent=args.mode == "coherent",
            synthesis=options,
        )
        report = {
            "configuration": circuit.metadata,
            "versions": {
                "python": python_version(),
                "qiskit": version("qiskit"),
                "numpy": version("numpy"),
            },
            "native": {
                "qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "gates": len(circuit.data),
                "gate_counts": dict(circuit.count_ops()),
            },
            "transpilation": None,
            "validation": None,
        }
        if args.simulate:
            from mixed_radix_qft.validation.fourier import verify_qft

            report["validation"] = verify_qft(circuit, seed=args.seed, max_states=args.max_states)
        exported = circuit
        if args.basis != "native":
            exported = transpile_circuit(
                circuit,
                basis=args.basis,
                optimization_level=args.optimization_level
                if args.optimization_level is not None
                else 1,
                seed=args.seed,
            )
            report["transpilation"] = {
                **exported.metadata["transpilation"],
                "depth": exported.depth(),
                "gates": len(exported.data),
                "gate_counts": dict(exported.count_ops()),
            }
        if args.output:
            export_qasm(exported, args.output / "circuit.qasm")
            (args.output / "metadata.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(report, indent=2))
        if args.draw:
            print(exported.draw(output="text", fold=100))
        if report["validation"] and not report["validation"]["passed"]:
            raise SystemExit(1)
    except (ValueError, MemoryError, OSError) as exc:
        cli.error(str(exc))
