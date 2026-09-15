# Article correspondence

The article's prime-factor and bounded-Hamming-weight assumptions are enforced by
the algorithm entry points. Functions shared with the thesis are retained only
when the selected article constructions or their validation need them.

| Article definition | Module | Implementation detail |
| --- | --- | --- |
| `F_m=C† (tensor F_mj) A C`, background / mixed-radix equation | `algorithms/general.py` | General coherent or terminal composition |
| `C` | `circuits/c.py` | Ordinary CRT residues; inverse works on every valid residue tuple |
| `A_j` | `circuits/a.py` | Multiplication by the Good–Thomas correction |
| Local `F_mj` | `circuits/local_qft.py`, `local_qft_components/` | Explicit Mosca–Zalka construction; Hadamard for modulus 2 |
| `T_j` | `circuits/counters.py` | Periodic classes and balanced count trees |
| `L_j^(A)` | `circuits/contributions.py` | Count-to-weighted-contribution lookup |
| `W_j` | `circuits/weighted_sum.py`, `circuits/adders/qfa2.py` | Carry-save compression and final Sklansky sum |
| `R_j` | `circuits/modular_reduction.py` | Threshold predicates and controlled subtraction |
| `U_j^(A)=R_j W_j L_j^(A) T_j` | `circuits/residues.py` | Reversible local computation with retained intermediate words |
| `C_res^(A)` | `circuits/residues.py` | Compute-copy-uncompute on private input copies |
| `C_sparse^(A)` | `circuits/sparse_conversion.py` | Erase input by the promised-domain decoder |
| Terminal Fourier output | `algorithms/sparse.py`, `classical/crt.py` | Ordinary CRT labels, followed by classical reconstruction after measurement |
| Coherent completion | `algorithms/sparse.py`, `circuits/c.py` | General `C†` appended after local QFTs |

## Selectable implementation variants

The source code includes additional realizations of the same promised transformation:
Cuccaro weighted-sum trees, arithmetic input decoders, projected-key lookup,
prefix modular reduction, deferred cleanup and several explicit local-arithmetic
backends. They remain selectable and are reported separately; they are not all
claimed to be the circuit realization described in the article text.

In particular, the in-place QFA2 implementation preserves two inputs and overwrites
the third with the parity word while writing a fresh carry word. Its inverse recovers
the original operands. The article presents the sum/carry identity; workspace
accounting here follows the emitted reversible implementation.

## Excluded material

- Thesis presets and command-line aliases.
- Standalone approximate arbitrary-order QFT and its certification machinery.
- Standalone Cleve–Watrous, Schönhage–Strassen and unrelated multiplier experiments.
- Historical resource sweeps, probe runners, plotting/Julia pipelines and datasets.
- Dense local-unitary and modular-permutation backends, and opaque skeleton circuits.
- Manuscript sources, PDFs, presentations, existing Git history and user credentials.

Dense arrays occur only in small validation references. Algorithm construction has
no dependency on the old `experiments` package or the new validation package.
