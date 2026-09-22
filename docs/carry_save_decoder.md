# Carry-save integration in the input decoder

Implemented and checked on 17 September 2026. Select
`decoder="crt-wallace-kogge-stone"` through the sparse QFT API or
`--decoder crt-wallace-kogge-stone` through the CLI. The default remains
`lookup`; `crt-kogge-stone` retains its original prefix-adder trees.
This integration is in this standalone package, not the thesis repository's
historical `experiments/` implementation.

## Construction and correctness

The decoder reads the fused labels `eta_j(x)=g_j*x mod m_j`, with
`M_j=m/m_j` and `g_j=M_j^(-1) mod m_j`. For valid fields it computes
`S=sum_j M_j*y_j`, then `r=S mod m`. Thus `r=x` for `y=eta(x), 0<=x<m`.
It XORs r into X and reverses all auxiliary computation, preserving Y.

The integration reuses `linear_sum.raw_weighted_sum` twice:

1. Generate the controlled constant words and compress them with a
   Wallace/QFA2 tree. One Kogge–Stone sum converts the final two words
   to the binary S register.
2. Fan out S coherently to separate constant comparators. Compute the
   threshold flags `[S >= ell*m]` in parallel.
3. Sum the selected copies of m through a second Wallace/QFA2 tree and
   final Kogge–Stone adder, obtaining `floor(S/m)*m`.
4. Subtract with the existing Kogge–Stone difference, XOR into X, and
   reverse the complete decoder computation.

Only private partial words are modified by the compressors. The public
Y register remains unchanged, so outer uncomputation remains valid after
X is updated. There are no measurements or resets. Gates are X/CX/CCX.

The width is chosen from `S_max=sum_j (m_j-1)*M_j`; valid sums and the
selected multiple of m fit in B=`bit_length(S_max)`. Intermediate
compression preserves the sum modulo `2**B`; since the valid total fits,
the final result is the exact integer sum. Outside valid field encodings,
the same modular arithmetic and bounded thresholds define a reversible
extension; that extension agrees with the old decoder, rather than the
sparse lookup's extension to zero.

The reference for tree-based QCSA and QFA2 is
[Kim et al., Tree-based Quantum Carry-Save Adder (2025), ePrint 2025/626](https://eprint.iacr.org/2025/626).
This CRT composition and the following bounds are derived from the code;
the paper is not claimed to prove this particular decoder.

## Logical resource bounds

Let n=`ceil(log2(m))`, q_Y be the total field width, and k the number of
moduli. Then q_Y=Theta(n), k<=n, `S_max<k*m`, B=O(n), and the number
of threshold flags is at most k-1.

With bounded-arity gates, all-to-all logical connectivity and enough
clean ancillas, the stages have the following bounds:

| Stage | Depth | Gates and allocated workspace, upper bound |
| --- | --- | --- |
| Controlled partials and carry-save reduction | O(log B + log q_Y) | O(q_Y B) |
| First Kogge–Stone sum | O(log B) | O(B log B) |
| Fan-out and parallel threshold comparison | O(log k + log B) | O(k B) |
| Quotient-multiple carry-save sum and final addition | O(log k + log B) | O(k B + B log B) |
| Final difference | O(log B) | O(B log B) |
| XOR and complete uncompute | Same asymptotic depth as forward computation | Same asymptotic size |

Logs are understood as `log(2+t)` for degenerate one-operand cases.
Consequently the **clean decoder** has D=O(log n), G=O(n²), and
Q=O(n²). The existing prefix-tree decoder has D=O(log² n) and the
conservative G,Q=O(n² log n) bounds. The derivation includes the fan-out,
comparison and cleanup stages; it does not identify Toffoli-depth with
total depth. A fixed decomposition of Toffoli into one/two-qubit gates
preserves these orders before physical routing.

These are upper bounds on this construction. They do not establish the
depth of the entire QFT: residue preparation and local Fourier transforms
must be counted separately. They also do not establish superiority over
the parallel sparse lookup, which is not the baseline in this experiment.

## Measurements

[Raw results](carry_save_decoder_results.json) record versions, factors,
gate counts and omitted compilations. Both variants use identical
residue settings (`w=2`, `wallace-qfa2`, `prefix`, `deferred`). Compilation
uses u/cx, optimization level 0, seed 271828, all-to-all connectivity.
Every decoder measurement includes XOR and full cleanup, not only a
forward arithmetic block. Qubit counts include the public registers and
all allocated workspace; they are not an optimized peak-memory schedule.

| n | Decoder native depth, old → new | Decoder u/cx depth, old → new | Decoder qubits, old → new |
| ---: | ---: | ---: | ---: |
| 8 | 222 → 184 | 1412 → 995 | 1025 → 535 |
| 19 | 372 → 280 | 2547 → 1445 | 7004 → 2358 |
| 33 | 456 → 324 | 3206 → 1673 | 21338 → 5840 |
| 65 | 612 → 396 | old omitted; new 2040 | 89002 → 19302 |

At n=33 this is a 47.82% reduction in compiled decoder depth and a
72.63% reduction in allocated qubits. For the full input conversion,
compiled depth decreases from 3878 to 2346 (39.50%).

The complete terminal QFT was also constructed at n=8 (m=210): its
compiled depth decreases from 10488 to 10072 (3.97%), with the same local
QFT synthesis. This illustrates why a large decoder improvement does not
translate into the same percentage for the entire transform. No state
simulation at m=210 was performed for these resource measurements.

Eighteen circuits were built. Fifteen were compiled; the old decoder and
both full conversions at n=65 exceeded the 150000-native-gate compilation
cap. Their compiled figures are explicitly absent, with no substituted
backend. The small-instance correctness tests below are separate from
these resource measurements. No hardware execution or routing was used.

Reproduce from the package root (choose a new result path):

```bash
.venv/bin/python examples/compare_crt_decoders.py --output /tmp/crt-comparison.json
.venv/bin/python -m mixed_radix_qft sparse --moduli 2 3 --w 2 \
  --decoder crt-wallace-kogge-stone --simulate
```

## Correctness checks

The complete package suite passed: **39 tests in 54.529 seconds**, with
no failures or skips. Ruff lint, formatting and `git diff --check` passed.

`tests/test_carry_save_decoder.py` checks:

- Every valid tuple for m=2,3,6,15,30,210, with several nonzero XOR
  targets. A direct integer formula is the oracle, and all physical
  output bits participate in each comparison, including clean workspace.
- Every physical Y encoding for m=3,6,15,30, including unused encodings:
  equality with the old decoder and involution on a nonzero target.
- Complex superpositions through the fused conversion at m=30,w=2,
  zero input/work registers at the output, and the inverse circuit.
- Terminal and coherent QFT against an independent DFT at m=6,
  w=0,1,2,3, including relative phases and workspace amplitudes.
- CLI selection, metadata and seeded DFT verification, and rejection
  of unsupported backend combinations.

The existing coherent-QFT regression also includes the new decoder with
both weighted-sum backends and w=0,1,2. Finite tests validate these instances;
they are not proofs of asymptotic depth or of hardware performance.
