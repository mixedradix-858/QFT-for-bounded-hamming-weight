# Attribution and source provenance

This repository extracts and reorganizes the Python/Qiskit implementation associated
with *Low-Depth Quantum Fourier Transform Circuits for Bounded-Hamming-Weight Inputs*.

Included source families are `general_mixed_radix.py`, `qiskit_sparse_qft.py`,
`mosca_zalka.py`, `explicit_primitives.py`, `parallel_arithmetic.py`,
`bounded_reduction.py`, `projected_lookup.py` and selected validation helpers/tests.
See `docs/source-map.md` for symbol-level provenance.

The mathematical constructions retain their original references:

- Cleve and Watrous, *Fast Parallel Circuits for the Quantum Fourier Transform*
  (mixed-radix decomposition and general CRT conversion).
- Good–Thomas CRT Fourier decomposition and correction factors.
- Mosca and Zalka, *Exact Quantum Fourier Transforms and Discrete Logarithm
  Algorithms* (local exact Fourier construction, estimation and amplification).
- Cuccaro et al., ripple-carry addition. The code calls Qiskit's
  `adder_ripple_c04` implementation; Qiskit remains an external dependency.
- Sklansky and Kogge–Stone prefix networks; Wang et al. for reversible prefix
  arithmetic as cited by the supplied article.
- Kim et al., tree-based quantum carry-save addition and the QFA2 compressor.
- Zhu et al., reversible lookup networks; Vandaele et al., threshold comparisons,
  as cited by the supplied article.

