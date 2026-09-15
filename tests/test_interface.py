"""Public API, complete QFTs, selected synthesis and isolated entry points."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from qiskit import QuantumCircuit

from mixed_radix_qft import SparseQFTConfig, SynthesisOptions, build_qft
from mixed_radix_qft.algorithms.sparse import build_sparse_qft
from mixed_radix_qft.circuits.local_qft import build_mosca_zalka_inplace_qft
from mixed_radix_qft.circuits.local_qft_components.modes import EXPLICIT_ARITHMETIC_MODES
from mixed_radix_qft.validation.fourier import independent_dft, sparse_error, verify_qft
from mixed_radix_qft.validation.simulation import simulate_sparse_state

ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = {"x", "cx", "ccx", "h", "p", "cp", "ry", "cry", "swap", "z"}


class InterfaceTests(unittest.TestCase):
    def test_each_local_synthesis_matches_dft_and_cleans_work(self):
        vector = np.array([1 + 2j, -2 + 1j, 3 - 1j]) / np.sqrt(20)
        expected = dict(enumerate(independent_dft(3) @ vector))
        for backend in sorted(EXPLICIT_ARITHMETIC_MODES):
            with self.subTest(backend=backend):
                circuit = build_mosca_zalka_inplace_qft(3, arithmetic=backend)
                actual = simulate_sparse_state(
                    circuit, dict(enumerate(vector)), prune_tolerance=1e-13
                )
                self.assertLess(sparse_error(actual, expected), 1e-8)
                self.assertFalse(set(circuit.count_ops()) - PRIMITIVES)

    def test_general_and_sparse_terminal_and_coherent(self):
        for algorithm in ("general", "sparse"):
            for coherent in (False, True):
                with self.subTest(algorithm=algorithm, coherent=coherent):
                    circuit = build_qft(
                        algorithm, (2, 3), w=2 if algorithm == "sparse" else None, coherent=coherent
                    )
                    self.assertTrue(verify_qft(circuit)["passed"])
                    self.assertFalse(set(circuit.count_ops()) - PRIMITIVES)

    def test_projected_and_reduction_variants_on_complex_states(self):
        for w in (0, 1, 3):
            for cleanup in ("local", "deferred"):
                for coherent in (False, True):
                    with self.subTest(w=w, cleanup=cleanup, coherent=coherent):
                        circuit = build_qft(
                            "sparse",
                            (2, 3),
                            w=w,
                            coherent=coherent,
                            synthesis=SynthesisOptions(
                                decoder="lookup-projected",
                                reduction_backend="prefix",
                                reduction_cleanup=cleanup,
                            ),
                        )
                        self.assertTrue(verify_qft(circuit)["passed"])

    def test_invalid_api_combinations_and_lookup_limit(self):
        for options in (
            SynthesisOptions(decoder="lookup"),
            SynthesisOptions(weighted_sum_backend="cuccaro"),
            SynthesisOptions(reduction_backend="prefix"),
            SynthesisOptions(reduction_cleanup="local"),
            SynthesisOptions(lookup_entry_limit=20),
        ):
            with self.assertRaises(ValueError):
                build_qft("general", (2, 3), synthesis=options)
        for kwargs in (
            {"algorithm": "general", "moduli": (2, 3), "w": 1},
            {"algorithm": "sparse", "moduli": (2, 3)},
            {"algorithm": "general", "moduli": (4, 3)},
            {"algorithm": "sparse", "moduli": (2, 3), "w": 4},
            {"algorithm": "sparse", "moduli": (2, 2), "w": 1},
            {"algorithm": "unknown", "moduli": (2, 3)},
            {
                "algorithm": "general",
                "moduli": (2, 3),
                "synthesis": SynthesisOptions(local_backend="lookup"),
            },
        ):
            with self.assertRaises(ValueError):
                build_qft(**kwargs)
        for decoder in ("lookup", "lookup-projected"):
            with self.assertRaises(MemoryError):
                build_qft(
                    "sparse",
                    (3, 5),
                    w=2,
                    synthesis=SynthesisOptions(decoder=decoder, lookup_entry_limit=1),
                )
        with self.assertRaises(ValueError):
            build_sparse_qft(SparseQFTConfig.from_moduli((3, 5), w=1, n=6))
        with self.assertRaises(ValueError):
            build_sparse_qft(SparseQFTConfig.from_moduli((2,), w=0), lookup_entry_limit=True)

    def test_cli_export_and_effective_synthesis(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            command = [
                sys.executable,
                str(ROOT / "main.py"),
                "sparse",
                "--moduli",
                "2",
                "3",
                "--w",
                "1",
                "--decoder",
                "lookup-projected",
                "--reduction",
                "prefix",
                "--cleanup",
                "deferred",
                "--basis",
                "u-cx",
                "--optimization-level",
                "0",
                "--simulate",
                "--output",
                str(output),
            ]
            process = subprocess.run(
                command, cwd=directory, capture_output=True, text=True, timeout=90
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertEqual(json.loads((output / "metadata.json").read_text()), report)
            self.assertTrue(report["validation"]["passed"])
            self.assertEqual(report["configuration"]["synthesis"]["decoder"], "lookup-projected")
            self.assertEqual(report["configuration"]["synthesis"]["reduction_cleanup"], "deferred")
            self.assertEqual(report["transpilation"]["basis"], ["u", "cx"])
            self.assertFalse(set(report["transpilation"]["gate_counts"]) - {"u", "cx"})
            source = (output / "circuit.qasm").read_text()
            self.assertTrue(source.startswith("OPENQASM 3.0;"))
            self.assertIn("qubit[", source)
            again = subprocess.run(
                command, cwd=directory, capture_output=True, text=True, timeout=20
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((output / "circuit.qasm").read_text(), source)

    def test_module_and_console_entry_points_outside_repository(self):
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with TemporaryDirectory() as directory:
            for entry in (
                [sys.executable, "-I", "-m", "mixed_radix_qft"],
                [str(Path(sys.executable).parent / "mixed-radix-qft")],
            ):
                process = subprocess.run(
                    [*entry, "general", "--moduli", "2", "--simulate"],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertTrue(json.loads(process.stdout)["validation"]["passed"])

    def test_cli_rejects_irrelevant_options(self):
        for args in (
            ["general", "--moduli", "2", "--w", "1"],
            ["general", "--moduli", "2", "--decoder", "lookup"],
            ["general", "--moduli", "2", "--optimization-level", "1"],
            ["sparse", "--moduli", "2"],
            ["sparse", "--moduli", "2", "--w", "0", "--max-states", "0"],
        ):
            process = subprocess.run(
                [sys.executable, "-m", "mixed_radix_qft", *args],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(process.returncode, 2)
            self.assertIn("error:", process.stderr)

    def test_physical_leakage_and_relative_phase_errors_fail(self):
        circuit = build_qft("general", (2,))
        corrupted = circuit.copy()
        corrupted.z(0)
        self.assertFalse(verify_qft(corrupted)["passed"])
        leaked = circuit.copy()
        leaked.x(leaked.num_qubits - 1)
        self.assertFalse(verify_qft(leaked)["passed"])

    def test_simulation_cap(self):
        circuit = QuantumCircuit(3)
        circuit.h(range(3))
        with self.assertRaises(MemoryError):
            simulate_sparse_state(circuit, {0: 1}, max_states=4)

    def test_core_imports_have_no_legacy_or_validation_dependencies(self):
        package = ROOT / "src/mixed_radix_qft"
        for directory in ("circuits", "algorithms", "classical"):
            for path in (package / directory).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse(
                            (node.module or "").startswith(
                                ("experiments", "tests", "mixed_radix_qft.validation")
                            ),
                            str(path),
                        )
                    if isinstance(node, ast.Name):
                        self.assertNotEqual(node.id, "UnitaryGate", str(path))
