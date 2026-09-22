"""Compare clean CRT decoders and matched sparse conversions, without statevectors.

Run with the package installed; --output must name a new JSON file. Both
backends use the same w=2, residue arithmetic and local QFT. Transpilation
uses u/cx, optimization 0, seed 271828, and all-to-all connectivity.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from qiskit import QuantumCircuit

from mixed_radix_qft.algorithms.sparse import build_sparse_qft
from mixed_radix_qft.circuits.decoders import _append_crt_inverse_decoder_xor
from mixed_radix_qft.circuits.sparse_conversion import build_architecture_fused_conversion
from mixed_radix_qft.config import SparseQFTConfig
from mixed_radix_qft.synthesis import transpile_circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transpile-cap", type=int, default=150000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    factors = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    rows = []
    for count in (4, 7, 10, 16):
        config = SparseQFTConfig.from_moduli(factors[:count], w=2)
        scopes = ("decoder", "conversion", "terminal") if count == 4 else ("decoder", "conversion")
        for scope in scopes:
            for decoder in ("crt-kogge-stone", "crt-wallace-kogge-stone"):
                options = dict(
                    decoder=decoder,
                    weighted_sum_backend="wallace-qfa2",
                    reduction_backend="prefix",
                    reduction_cleanup="deferred",
                )
                if scope == "decoder":
                    circuit = QuantumCircuit(config.q_y + config.n)
                    _append_crt_inverse_decoder_xor(
                        circuit,
                        circuit.qubits[: config.q_y],
                        circuit.qubits[config.q_y :],
                        config,
                        prefix="kogge-stone",
                        sum_backend="wallace-qfa2" if "wallace" in decoder else "prefix-tree",
                    )
                elif scope == "conversion":
                    circuit = build_architecture_fused_conversion(
                        config, explicit_primitives=True, **options
                    )
                else:
                    circuit = build_sparse_qft(config, coherent=False, **options)
                row = dict(
                    n=config.n,
                    moduli=list(config.moduli),
                    w=config.w,
                    scope=scope,
                    decoder=decoder,
                    qubits=circuit.num_qubits,
                    native_gates=len(circuit.data),
                    native_depth=circuit.depth(),
                    compiled_gates=None,
                    compiled_depth=None,
                    compilation="omitted: native gate cap",
                )
                if len(circuit.data) <= args.transpile_cap:
                    compiled = transpile_circuit(
                        circuit, basis="u-cx", optimization_level=0, seed=271828
                    )
                    row.update(
                        compiled_gates=len(compiled.data),
                        compiled_depth=compiled.depth(),
                        compilation="completed",
                    )
                    del compiled
                rows.append(row)
                print(json.dumps(row), flush=True)
                del circuit
    report = dict(
        versions=dict(python=python_version(), qiskit=version("qiskit"), numpy=version("numpy")),
        basis=["u", "cx"],
        optimization_level=0,
        seed=271828,
        topology="all-to-all",
        transpile_cap=args.transpile_cap,
        weighted_sum="wallace-qfa2",
        reduction="prefix",
        cleanup="deferred",
        local_backend="explicit-register-sklansky",
        rows=rows,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
