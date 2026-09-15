"""Focused regression tests retained from the article implementation."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector

from mixed_radix_qft.circuits.adders.cuccaro import (
    append_controlled_cuccaro_add,
    append_cuccaro_add,
)
from mixed_radix_qft.circuits.adders.fourier import build_fourier_controlled_constant_additions
from mixed_radix_qft.circuits.adders.kogge_stone import (
    append_kogge_stone_difference,
    append_kogge_stone_sum,
    kogge_stone_prefix_workspace_size,
)
from mixed_radix_qft.circuits.adders.qfa2 import qfa2_word_compressor
from mixed_radix_qft.circuits.adders.sklansky import (
    append_sklansky_difference,
    append_sklansky_less_than,
    append_sklansky_sum,
    sklansky_fixed_adder,
    sklansky_half_adder,
    sklansky_prefix_workspace_size,
)
from mixed_radix_qft.circuits.comparators import (
    append_log_depth_constant_comparator,
    comparator_workspace_size,
)
from mixed_radix_qft.circuits.controls import append_log_depth_mcx, mcx_workspace_size
from mixed_radix_qft.circuits.local_qft_components.constant_arithmetic import (
    build_controlled_modular_constant_add,
)
from mixed_radix_qft.circuits.local_qft_components.register_arithmetic import (
    build_modular_register_operation,
)
from mixed_radix_qft.circuits.local_qft_components.rounding import build_explicit_rounding_circuit
from mixed_radix_qft.circuits.weighted_sum import (
    build_wallace_qfa2_sum_circuit,
    wallace_qfa2_reduction_shape,
)
from mixed_radix_qft.classical.estimator_parameters import accepted_phase_value, rounded_estimate
from mixed_radix_qft.validation.reference import expected_fourier_state, require_dense_memory
from mixed_radix_qft.validation.simulation import (
    simulate_classical_reversible_circuit,
    simulate_sparse_state,
)


class ArithmeticTests(unittest.TestCase):
    def test_clean_sklansky_adders_and_comparator(self) -> None:
        """Verify clean sklansky adders and comparator."""
        for width in range(1, 6):
            mask = (1 << width) - 1
            fixed = sklansky_fixed_adder(width)
            half = sklansky_half_adder(width)
            comparator_work = sklansky_prefix_workspace_size(width)
            comparator = QuantumCircuit(2 * width + 1 + comparator_work)
            append_sklansky_less_than(
                comparator,
                comparator.qubits[:width],
                comparator.qubits[width : 2 * width],
                comparator.qubits[2 * width],
                comparator.qubits[2 * width + 1 :],
            )
            for left in range(1 << width):
                for right in range(1 << width):
                    initial = [
                        *(left >> bit & 1 for bit in range(width)),
                        *(right >> bit & 1 for bit in range(width)),
                    ]
                    fixed_result = simulate_classical_reversible_circuit(fixed, initial)
                    half_result = simulate_classical_reversible_circuit(half, initial)
                    compare_result = simulate_classical_reversible_circuit(comparator, initial)
                    observed_sum = sum((fixed_result[width + bit] << bit for bit in range(width)))
                    self.assertEqual(observed_sum, left + right & mask)
                    self.assertFalse(any(fixed_result[2 * width :]))
                    self.assertEqual(half_result[2 * width], left + right >> width)
                    self.assertFalse(any(half_result[2 * width + 1 :]))
                    self.assertEqual(compare_result[2 * width], int(left < right))
                    self.assertFalse(any(compare_result[2 * width + 1 :]))

    def test_clean_out_of_place_prefix_sum_and_difference(self) -> None:
        """Verify clean out of place prefix sum and difference."""
        implementations = (
            (
                "sklansky",
                sklansky_prefix_workspace_size,
                append_sklansky_sum,
                append_sklansky_difference,
            ),
            (
                "kogge-stone",
                kogge_stone_prefix_workspace_size,
                append_kogge_stone_sum,
                append_kogge_stone_difference,
            ),
        )
        for name, workspace_size, append_sum, append_difference in implementations:
            for width in range(1, 6):
                with self.subTest(prefix=name, width=width):
                    mask = (1 << width) - 1
                    workspace_width = workspace_size(width)
                    addition = QuantumCircuit(3 * width + workspace_width)
                    subtraction = QuantumCircuit(3 * width + workspace_width)
                    append_sum(
                        addition,
                        addition.qubits[:width],
                        addition.qubits[width : 2 * width],
                        addition.qubits[2 * width : 3 * width],
                        addition.qubits[3 * width :],
                    )
                    append_difference(
                        subtraction,
                        subtraction.qubits[:width],
                        subtraction.qubits[width : 2 * width],
                        subtraction.qubits[2 * width : 3 * width],
                        subtraction.qubits[3 * width :],
                    )
                    for left in range(1 << width):
                        for right in range(1 << width):
                            initial = [
                                *(left >> bit & 1 for bit in range(width)),
                                *(right >> bit & 1 for bit in range(width)),
                            ]
                            added = simulate_classical_reversible_circuit(addition, initial)
                            subtracted = simulate_classical_reversible_circuit(subtraction, initial)
                            observed_sum = sum(
                                (added[2 * width + bit] << bit for bit in range(width))
                            )
                            observed_difference = sum(
                                (subtracted[2 * width + bit] << bit for bit in range(width))
                            )
                            self.assertEqual(observed_sum, left + right & mask)
                            self.assertEqual(observed_difference, left - right & mask)
                            self.assertFalse(any(added[3 * width :]))
                            self.assertFalse(any(subtracted[3 * width :]))

    def test_sklansky_modular_register_arithmetic(self) -> None:
        """Verify sklansky modular register arithmetic."""
        prime = 3
        width = (prime - 1).bit_length()
        for operation in ("add", "phase-add", "subtract"):
            circuit = build_modular_register_operation(prime, operation, adder="sklansky")
            source_values = range(1 << width) if operation == "phase-add" else range(prime)
            for source in source_values:
                for target in range(prime):
                    initial_index = source | target << width
                    final = simulate_sparse_state(circuit, {initial_index: 1})
                    expected_target = (
                        (target - source) % prime
                        if operation == "subtract"
                        else (target + source) % prime
                    )
                    expected_index = source | expected_target << width
                    self.assertEqual(set(final), {expected_index})
                    self.assertAlmostEqual(abs(final[expected_index]), 1.0, places=10)
        phase_add = build_modular_register_operation(prime, "phase-add", adder="sklansky")
        for phase in range(1 << width):
            for label in range(prime):
                fourier = expected_fourier_state(prime, label)
                initial = {
                    phase | value << width: amplitude
                    for value, amplitude in enumerate(fourier)
                    if abs(amplitude) > 1e-12
                }
                actual = simulate_sparse_state(phase_add, initial)
                expected_phase = np.exp(-2j * np.pi * label * phase / prime)
                support = set(actual) | set(initial)
                error = np.sqrt(
                    sum(
                        (
                            abs(actual.get(index, 0) - expected_phase * initial.get(index, 0)) ** 2
                            for index in support
                        )
                    )
                )
                self.assertLess(error, 1e-10)


class ExplicitPrimitiveTests(unittest.TestCase):
    def test_qfa2_word_compressor_truth_table_and_basis_depth(self) -> None:
        """Verify qfa2 word compressor truth table and basis depth."""
        width = 3
        compressor = qfa2_word_compressor(width)
        self.assertEqual(compressor.depth(), 5)
        self.assertEqual(compressor.count_ops().get("ccx", 0), width - 1)
        for first in range(1 << width):
            for second in range(1 << width):
                for third in range(1 << width):
                    initial = [
                        *(first >> bit & 1 for bit in range(width)),
                        *(second >> bit & 1 for bit in range(width)),
                        *(third >> bit & 1 for bit in range(width)),
                    ]
                    final = simulate_classical_reversible_circuit(compressor, initial)
                    observed_first = sum((final[bit] << bit for bit in range(width)))
                    observed_second = sum((final[width + bit] << bit for bit in range(width)))
                    sum_word = sum((final[2 * width + bit] << bit for bit in range(width)))
                    carry_word = sum((final[3 * width + bit] << bit for bit in range(width)))
                    self.assertEqual(observed_first, first)
                    self.assertEqual(observed_second, second)
                    self.assertEqual(
                        (sum_word + carry_word) % (1 << width),
                        (first + second + third) % (1 << width),
                    )
                    self.assertEqual(final[3 * width], 0)
        basis = transpile(qfa2_word_compressor(14), basis_gates=["u", "cx"], optimization_level=0)
        self.assertEqual(basis.depth(), 14)
        self.assertEqual(basis.size(), 262)

    def test_wallace_qfa2_sum_and_clean_inverse(self) -> None:
        """Verify wallace qfa2 sum and clean inverse."""
        self.assertEqual(wallace_qfa2_reduction_shape(4098), (20, 4096))
        width = 3
        operands = 5
        block, output_indices = build_wallace_qfa2_sum_circuit(operands, width)
        mask = (1 << width) - 1
        values_to_test = ((0, 0, 0, 0, 0), (1, 2, 3, 4, 5), (7, 7, 7, 7, 7), (6, 1, 5, 2, 4))
        for values in values_to_test:
            initial = [value >> bit & 1 for value in values for bit in range(width)]
            final = simulate_classical_reversible_circuit(block, initial)
            observed = sum((final[index] << bit for bit, index in enumerate(output_indices)))
            self.assertEqual(observed, sum(values) & mask)
        copied = QuantumRegister(width, "copied_sum")
        clean = QuantumCircuit(*block.qregs, copied)
        clean.compose(block, inplace=True)
        for bit, source_index in enumerate(output_indices):
            clean.cx(clean.qubits[source_index], copied[bit])
        clean.compose(block.inverse(), qubits=clean.qubits[: block.num_qubits], inplace=True)
        for values in values_to_test:
            initial = [value >> bit & 1 for value in values for bit in range(width)]
            final = simulate_classical_reversible_circuit(clean, initial)
            self.assertEqual(final[: operands * width], initial)
            self.assertFalse(any(final[operands * width : block.num_qubits]))
            copied_value = sum((final[block.num_qubits + bit] << bit for bit in range(width)))
            self.assertEqual(copied_value, sum(values) & mask)

    def test_balanced_mcx_all_control_states(self) -> None:
        """Verify balanced mcx all control states."""
        from qiskit import QuantumCircuit, QuantumRegister

        controls = QuantumRegister(5, "controls")
        target = QuantumRegister(1, "target")
        work = QuantumRegister(mcx_workspace_size(5), "work")
        for control_state in (0, 1, 7, 19, 31):
            circuit = QuantumCircuit(controls, target, work)
            append_log_depth_mcx(circuit, controls, target[0], work, ctrl_state=control_state)
            for basis in range(1 << 5):
                require_dense_memory(circuit.num_qubits)
                final = Statevector.from_int(basis, 1 << circuit.num_qubits).evolve(circuit)
                expected = basis
                if basis == control_state:
                    expected |= 1 << 5
                self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)

    def test_cuccaro_add_and_controlled_add(self) -> None:
        """Verify cuccaro add and controlled add."""
        from qiskit import QuantumCircuit, QuantumRegister

        width = 3
        source = QuantumRegister(width, "source")
        target = QuantumRegister(width, "target")
        helper = QuantumRegister(1, "helper")
        circuit = QuantumCircuit(source, target, helper)
        append_cuccaro_add(circuit, source, target, helper[0])
        for left in range(1 << width):
            for right in range(1 << width):
                initial = left | right << width
                expected = left | (left + right) % (1 << width) << width
                require_dense_memory(circuit.num_qubits)
                final = Statevector.from_int(initial, 1 << circuit.num_qubits).evolve(circuit)
                self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)
        control = QuantumRegister(1, "control")
        source = QuantumRegister(width, "c_source")
        target = QuantumRegister(width, "c_target")
        helper = QuantumRegister(1, "c_helper")
        work = QuantumRegister(2, "c_work")
        controlled = QuantumCircuit(control, source, target, helper, work)
        append_controlled_cuccaro_add(controlled, control[0], source, target, helper[0], work)
        for control_value in (0, 1):
            for left in range(1 << width):
                for right in range(1 << width):
                    initial = control_value | left << 1 | right << 1 + width
                    output_right = (left + right) % (1 << width) if control_value else right
                    expected = control_value | left << 1 | output_right << 1 + width
                    require_dense_memory(controlled.num_qubits)
                    final = Statevector.from_int(initial, 1 << controlled.num_qubits).evolve(
                        controlled
                    )
                    self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)

    def test_log_depth_constant_comparator(self) -> None:
        """Verify log depth constant comparator."""
        from qiskit import QuantumCircuit, QuantumRegister

        width = 3
        state = QuantumRegister(width, "state")
        flag = QuantumRegister(1, "flag")
        work = QuantumRegister(comparator_workspace_size(width), "work")
        for threshold in (0, 1, 3, 5, 7):
            circuit = QuantumCircuit(state, flag, work)
            append_log_depth_constant_comparator(circuit, state, flag[0], threshold, work)
            for value in range(1 << width):
                require_dense_memory(circuit.num_qubits)
                final = Statevector.from_int(value, 1 << circuit.num_qubits).evolve(circuit)
                expected = value
                if value >= threshold:
                    expected |= 1 << width
                self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)


class MoscaZalkaTests(unittest.TestCase):
    def test_explicit_controlled_modular_addition(self) -> None:
        """Verify explicit controlled modular addition."""
        for prime in (3, 5):
            width = (prime - 1).bit_length()
            for value in (-2, 1, 2):
                circuit = build_controlled_modular_constant_add(prime, value)
                self.assertLessEqual(set(circuit.count_ops()), {"x", "cx", "ccx"})
                for control in (0, 1):
                    for target in range(prime):
                        initial = [control] + [target >> bit & 1 for bit in range(width)]
                        final = simulate_classical_reversible_circuit(circuit, initial)
                        observed = sum((final[1 + bit] << bit for bit in range(width)))
                        expected = (target + value) % prime if control else target
                        self.assertEqual(observed, expected)
                        self.assertFalse(any(final[1 + width :]))

    def test_shared_fourier_constant_additions(self) -> None:
        """Verify shared fourier constant additions."""
        values = (-3, 5)
        for width in (2, 3):
            circuit = build_fourier_controlled_constant_additions(width, values)
            control_width = len(values)
            order = 1 << width
            self.assertLessEqual(set(circuit.count_ops()), {"h", "cp", "swap"})
            for controls in range(1 << control_width):
                increment = sum((value for bit, value in enumerate(values) if controls >> bit & 1))
                for target in range(order):
                    initial = controls | target << control_width
                    expected_target = (target + increment) % order
                    expected = controls | expected_target << control_width
                    require_dense_memory(circuit.num_qubits)
                    final = Statevector.from_int(initial, 1 << circuit.num_qubits).evolve(circuit)
                    self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)
        for prime in (3, 5, 7):
            for value in (-2, 1, 2):
                circuit = build_controlled_modular_constant_add(prime, value, addition="fourier")
                for control in (0, 1):
                    for target in range(prime):
                        initial = control | target << 1
                        expected_target = (target + value) % prime if control else target
                        expected = control | expected_target << 1
                        require_dense_memory(circuit.num_qubits)
                        final = Statevector.from_int(initial, 1 << circuit.num_qubits).evolve(
                            circuit
                        )
                        self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)
        prime = 3
        width = (prime - 1).bit_length()
        value = 2
        shift = value % prime
        history = build_controlled_modular_constant_add(
            prime, value, addition="fourier", preserve_predicate=True
        )
        predicate_index = 1 + width
        for control in (0, 1):
            for target in range(prime):
                initial = control | target << 1
                expected_target = (target + value) % prime if control else target
                expected = control | expected_target << 1
                if target >= prime - shift:
                    expected |= 1 << predicate_index
                require_dense_memory(history.num_qubits)
                final = Statevector.from_int(initial, 1 << history.num_qubits).evolve(history)
                self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)
        round_trip = history.compose(history.inverse())
        for control in (0, 1):
            for target in range(prime):
                initial = control | target << 1
                require_dense_memory(round_trip.num_qubits)
                final = Statevector.from_int(initial, 1 << round_trip.num_qubits).evolve(round_trip)
                self.assertAlmostEqual(abs(final.data[initial]), 1.0, places=10)

    def test_clean_modular_register_arithmetic_and_phase_kickback(self) -> None:
        """Verify clean modular register arithmetic and phase kickback."""
        prime = 3
        width = (prime - 1).bit_length()
        for operation in ("add", "phase-add", "subtract"):
            circuit = build_modular_register_operation(prime, operation)
            source_values = range(1 << width) if operation == "phase-add" else range(prime)
            for source in source_values:
                for target in range(prime):
                    initial = source | target << width
                    expected_target = (
                        (target - source) % prime
                        if operation == "subtract"
                        else (target + source) % prime
                    )
                    expected = source | expected_target << width
                    require_dense_memory(circuit.num_qubits)
                    final = Statevector.from_int(initial, 1 << circuit.num_qubits).evolve(circuit)
                    self.assertAlmostEqual(abs(final.data[expected]), 1.0, places=10)
        phase_add = build_modular_register_operation(prime, "phase-add")
        require_dense_memory(phase_add.num_qubits)
        dimension = 1 << phase_add.num_qubits
        for phase in range(1 << width):
            for label in range(prime):
                initial = np.zeros(dimension, dtype=complex)
                fourier = expected_fourier_state(prime, label)
                for value, amplitude in enumerate(fourier):
                    initial[phase | value << width] = amplitude
                actual = Statevector(initial).evolve(phase_add).data
                expected_phase = np.exp(-2j * np.pi * label * phase / prime)
                self.assertLess(np.linalg.norm(actual - expected_phase * initial), 1e-10)

    def test_explicit_rounding_and_filter(self) -> None:
        """Verify explicit rounding and filter."""
        for prime in (3, 5):
            width = (prime - 1).bit_length()
            order = 1 << width
            circuit = build_explicit_rounding_circuit(prime)
            offsets: dict[str, int] = {}
            cursor = 0
            for register in circuit.qregs:
                offsets[register.name] = cursor
                cursor += len(register)
            rounding_offset = offsets["rounding_work"]
            modular_offset = offsets["modular_work"]
            rounding_width = len(circuit.qregs[-1])
            for phase in range(order):
                initial = [phase >> bit & 1 for bit in range(width)]
                final = simulate_classical_reversible_circuit(circuit, initial)
                estimate = sum((final[width + bit] << bit for bit in range(width)))
                self.assertEqual(estimate, rounded_estimate(phase, prime, order))
                self.assertEqual(final[2 * width], accepted_phase_value(phase, prime, order))
                self.assertFalse(any(final[modular_offset:rounding_offset]))
                product = phase * prime
                observed_product = sum(
                    (final[rounding_offset + bit] << bit for bit in range(2 * width))
                )
                self.assertEqual(observed_product, product)
                allowed_dirty = {
                    *range(rounding_offset, rounding_offset + 2 * width),
                    rounding_offset + rounding_width - 1,
                }
                self.assertFalse(
                    any(
                        (
                            final[index] and index not in allowed_dirty
                            for index in range(rounding_offset, rounding_offset + rounding_width)
                        )
                    )
                )
