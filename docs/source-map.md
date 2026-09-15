# Source map

Extracted from the local article implementation; no source repository is required at runtime.

| New module | Original symbols |
| --- | --- |
| `algorithms.general` | `experiments/general_mixed_radix.py::build_general_mixed_qft` |
| `circuits.a` | `experiments/general_mixed_radix.py::build_general_a` |
| `circuits.adders.constant` | `experiments/explicit_primitives.py::append_controlled_constant_add` |
| `circuits.adders.cuccaro` | `experiments/explicit_primitives.py::cuccaro_fixed_adder`; `experiments/explicit_primitives.py::cuccaro_half_adder`; `experiments/explicit_primitives.py::cuccaro_full_adder`; `experiments/explicit_primitives.py::append_cuccaro_add`; `experiments/explicit_primitives.py::append_cuccaro_half_add`; `experiments/explicit_primitives.py::append_cuccaro_full_add`; `experiments/explicit_primitives.py::append_controlled_cuccaro_add` |
| `circuits.adders.dispatch` | `experiments/qiskit_sparse_qft.py::_append_modular_sum` |
| `circuits.adders.fourier` | `experiments/mosca_zalka.py::append_fourier_controlled_constant_additions`; `experiments/mosca_zalka.py::build_fourier_controlled_constant_additions` |
| `circuits.adders.kogge_stone` | `experiments/explicit_primitives.py::kogge_stone_prefix_workspace_size`; `experiments/explicit_primitives.py::_append_kogge_stone_prefix_compute`; `experiments/explicit_primitives.py::_append_kogge_stone_sum_xor`; `experiments/explicit_primitives.py::append_kogge_stone_sum`; `experiments/explicit_primitives.py::append_kogge_stone_difference` |
| `circuits.adders.prefix_support` | `experiments/explicit_primitives.py::_record_self_inverse_gate`; `experiments/explicit_primitives.py::_uncompute_recorded_gates`; `experiments/explicit_primitives.py::_append_balanced_fanout`; `experiments/explicit_primitives.py::_append_cx_swap` |
| `circuits.adders.qfa2` | `experiments/explicit_primitives.py::append_qfa2_word_compressor`; `experiments/explicit_primitives.py::qfa2_word_compressor` |
| `circuits.adders.sklansky` | `experiments/explicit_primitives.py::_sklansky_groups`; `experiments/explicit_primitives.py::sklansky_prefix_workspace_size`; `experiments/explicit_primitives.py::sklansky_inplace_workspace_size`; `experiments/explicit_primitives.py::_append_sklansky_prefix_compute`; `experiments/explicit_primitives.py::_append_sklansky_sum_xor`; `experiments/explicit_primitives.py::_append_clean_sklansky_add`; `experiments/explicit_primitives.py::sklansky_fixed_adder`; `experiments/explicit_primitives.py::sklansky_half_adder`; `experiments/explicit_primitives.py::append_sklansky_add`; `experiments/explicit_primitives.py::append_sklansky_sum`; `experiments/explicit_primitives.py::append_sklansky_difference`; `experiments/explicit_primitives.py::append_sklansky_half_add`; `experiments/explicit_primitives.py::append_sklansky_less_than` |
| `circuits.bounded_reduction` | `experiments/bounded_reduction.py::REDUCTION_BACKENDS`; `experiments/bounded_reduction.py::REDUCTION_CLEANUP_MODES`; `experiments/bounded_reduction.py::validate_reduction_options`; `experiments/bounded_reduction.py::_threshold_flags`; `experiments/bounded_reduction.py::_prefix_correction`; `experiments/bounded_reduction.py::append_bounded_reduction` |
| `circuits.c` | `experiments/general_mixed_radix.py::build_general_c` |
| `circuits.comparators` | `experiments/explicit_primitives.py::comparator_workspace_size`; `experiments/explicit_primitives.py::append_log_depth_constant_comparator` |
| `circuits.contributions` | `experiments/qiskit_sparse_qft.py::_append_contribution_lookups` |
| `circuits.controls` | `experiments/explicit_primitives.py::mcx_workspace_size`; `experiments/explicit_primitives.py::_normalized_ctrl_state`; `experiments/explicit_primitives.py::_toggle_zero_controls`; `experiments/explicit_primitives.py::append_log_depth_mcx`; `experiments/explicit_primitives.py::append_log_depth_controlled_ry`; `experiments/explicit_primitives.py::controlled_ry_workspace_size` |
| `circuits.counters` | `experiments/qiskit_sparse_qft.py::_append_counter_forest`; `experiments/qiskit_sparse_qft.py::build_counter_forest_circuit` |
| `circuits.decoders` | `experiments/qiskit_sparse_qft.py::_append_prefix_sum_tree`; `experiments/qiskit_sparse_qft.py::CRTInverseDecoderBlock`; `experiments/qiskit_sparse_qft.py::build_crt_inverse_decoder_block`; `experiments/qiskit_sparse_qft.py::_append_crt_inverse_decoder_xor` |
| `circuits.fanout` | `experiments/parallel_arithmetic.py::fanout`; `experiments/qiskit_sparse_qft.py::_fanout_to_targets`; `experiments/qiskit_sparse_qft.py::_uncompute_fanout`; `experiments/qiskit_sparse_qft.py::_append_controlled_binary_constant` |
| `circuits.linear_mod` | `experiments/parallel_arithmetic.py::build_linear_mod_xor` |
| `circuits.linear_sum` | `experiments/parallel_arithmetic.py::raw_weighted_sum` |
| `circuits.local_qft` | `experiments/mosca_zalka.py::build_mosca_zalka_qft`; `experiments/mosca_zalka.py::build_mosca_zalka_inplace_qft` |
| `circuits.local_qft_components.amplification` | `experiments/mosca_zalka.py::build_exact_amplified_estimator` |
| `circuits.local_qft_components.constant_arithmetic` | `experiments/mosca_zalka.py::ModularArithmeticWorkspace`; `experiments/mosca_zalka.py::modular_arithmetic_workspace_size`; `experiments/mosca_zalka.py::_partition_modular_workspace`; `experiments/mosca_zalka.py::append_controlled_modular_constant_add`; `experiments/mosca_zalka.py::build_controlled_modular_constant_add` |
| `circuits.local_qft_components.estimator` | `experiments/mosca_zalka.py::EstimatorRegisters`; `experiments/mosca_zalka.py::build_uniformized_estimator` |
| `circuits.local_qft_components.modes` | `experiments/mosca_zalka.py::EXPLICIT_ARITHMETIC_MODES`; `experiments/mosca_zalka.py::_uses_explicit_arithmetic`; `experiments/mosca_zalka.py::_addition_backend`; `experiments/mosca_zalka.py::_uses_history_predicates`; `experiments/mosca_zalka.py::_uses_register_arithmetic`; `experiments/mosca_zalka.py::_register_adder` |
| `circuits.local_qft_components.qfp` | `experiments/mosca_zalka.py::QFPLayout`; `experiments/mosca_zalka.py::build_qfp` |
| `circuits.local_qft_components.qfs` | `experiments/mosca_zalka.py::append_rephase`; `experiments/mosca_zalka.py::build_qfs` |
| `circuits.local_qft_components.range_preparation` | `experiments/mosca_zalka.py::_prefix_ctrl_state`; `experiments/mosca_zalka.py::range_preparation_workspace_size`; `experiments/mosca_zalka.py::append_range_preparation`; `experiments/mosca_zalka.py::build_range_preparation` |
| `circuits.local_qft_components.reflections` | `experiments/mosca_zalka.py::_append_multi_controlled_phase_flip`; `experiments/mosca_zalka.py::append_good_phase_oracle`; `experiments/mosca_zalka.py::append_zero_ancilla_phase_oracle` |
| `circuits.local_qft_components.register_arithmetic` | `experiments/mosca_zalka.py::RegisterModularWorkspace`; `experiments/mosca_zalka.py::register_modular_workspace_size`; `experiments/mosca_zalka.py::_partition_register_modular_workspace`; `experiments/mosca_zalka.py::append_register_less_than`; `experiments/mosca_zalka.py::append_modular_register_add`; `experiments/mosca_zalka.py::append_phase_modular_register_add`; `experiments/mosca_zalka.py::append_modular_register_subtract`; `experiments/mosca_zalka.py::build_modular_register_operation` |
| `circuits.local_qft_components.rounding` | `experiments/mosca_zalka.py::RoundingWorkspace`; `experiments/mosca_zalka.py::rounding_workspace_size`; `experiments/mosca_zalka.py::_partition_rounding_workspace`; `experiments/mosca_zalka.py::append_explicit_rounding`; `experiments/mosca_zalka.py::build_explicit_rounding_circuit` |
| `circuits.lookup` | `experiments/qiskit_sparse_qft.py::_append_serial_lookup_xor`; `experiments/qiskit_sparse_qft.py::_append_projected_decoder_xor`; `experiments/qiskit_sparse_qft.py::_xor_tree_into_target`; `experiments/qiskit_sparse_qft.py::_append_parallel_lookup_xor` |
| `circuits.modular_reduction` | `experiments/qiskit_sparse_qft.py::_append_threshold_reduction`; `experiments/qiskit_sparse_qft.py::build_threshold_reduction_circuit` |
| `circuits.residues` | `experiments/qiskit_sparse_qft.py::ResidueComputeBlock`; `experiments/qiskit_sparse_qft.py::build_residue_compute_block`; `experiments/qiskit_sparse_qft.py::_append_compute_copy_uncompute`; `experiments/qiskit_sparse_qft.py::build_architecture_c_res` |
| `circuits.sparse_conversion` | `experiments/qiskit_sparse_qft.py::build_architecture_fused_conversion` |
| `circuits.weighted_sum` | `experiments/qiskit_sparse_qft.py::_append_balanced_sum_tree`; `experiments/qiskit_sparse_qft.py::_append_wallace_qfa2_sum`; `experiments/qiskit_sparse_qft.py::build_wallace_qfa2_sum_circuit` |
| `circuits.workspace` | `experiments/parallel_arithmetic.py::fresh`; `experiments/parallel_arithmetic.py::undo`; `experiments/parallel_arithmetic.py::append_block` |
| `classical.crt` | `experiments/qiskit_sparse_qft.py::encode_fields`; `experiments/qiskit_sparse_qft.py::decode_fields`; `experiments/qiskit_sparse_qft.py::good_thomas_multipliers`; `experiments/qiskit_sparse_qft.py::transformed_crt_tuple`; `experiments/qiskit_sparse_qft.py::crt_inverse` |
| `classical.decoder_tables` | `experiments/qiskit_sparse_qft.py::fused_residue_entries`; `experiments/qiskit_sparse_qft.py::fused_decoder_entries`; `experiments/qiskit_sparse_qft.py::projected_decoder_plan` |
| `classical.estimator_parameters` | `experiments/mosca_zalka.py::rounded_estimate`; `experiments/mosca_zalka.py::accepted_phase_value`; `experiments/mosca_zalka.py::uniform_success_probability` |
| `classical.number_theory` | `experiments/experiment_support.py::validate_moduli`; `experiments/qiskit_sparse_qft.py::ceil_log2`; `experiments/qiskit_sparse_qft.py::is_prime` |
| `classical.projected_lookup` | `experiments/projected_lookup.py::LookupProjection`; `experiments/projected_lookup.py::project_key`; `experiments/projected_lookup.py::select_injective_bits`; `experiments/projected_lookup.py::project_lookup_table` |
| `classical.sparse_domain` | `experiments/qiskit_sparse_qft.py::counter_width`; `experiments/qiskit_sparse_qft.py::sparse_values`; `experiments/qiskit_sparse_qft.py::sparse_value_count`; `experiments/qiskit_sparse_qft.py::periodic_classes` |
| `config` | `experiments/qiskit_sparse_qft.py::WEIGHTED_SUM_BACKENDS`; `experiments/qiskit_sparse_qft.py::SparseQFTConfig` |
| `registers` | `experiments/general_mixed_radix.py::crt_layout` |

## New entry points and extraction adjustments

- `algorithms/sparse.py::build_sparse_qft` composes the extracted sparse conversion,
  selected explicit local QFTs and general `C†`. Unlike the original wrapper,
  it exposes every retained explicit local backend. Its default local construction
  is unchanged; the default input decoder follows the article (`lookup`).
- `synthesis.py::build_qft` validates applicable options and records the effective
  configuration. `transpile_circuit` handles basis compilation independently.
- `cli.py`, root `main.py`, package `__main__.py` and `export.py` provide the unified
  generation interface. They replace the source project's historical runners.
- The MZ estimator retains only its explicit arithmetic paths. Dense lookup-oracle
  branches were removed, preserving the emitted gates of all five retained modes.
- `circuits/counters.py::build_counter_forest_circuit` now exposes an
  `explicit_primitives` keyword, defaulting to `True`; the count-tree algorithm
  itself is unchanged.
- `circuits/weighted_sum.py::wallace_qfa2_reduction_shape` retains the original
  classical shape helper for component verification.
- `validation/reference.py` retains `experiment_support.require_dense_memory` and
  `mosca_zalka.expected_fourier_state` for small independent tests.
- `validation/simulation.py` contains the classical and bounded-support simulators
  from `test_qiskit_sparse_qft.py`. `validation/fourier.py` adapts the independent
  references from `test_validation_regressions.py` and adds the CLI check.

The focused regression tests originate from `test_qiskit_sparse_qft.py`,
`test_general_qft.py`, `test_bounded_reduction.py`, `test_projected_lookup.py` and
`test_validation_regressions.py`. New tests cover the public API, CLI, explicit
counter builder, failure detection and standalone installed entry points.
