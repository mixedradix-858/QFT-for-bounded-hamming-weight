# Validation record

Validated on Linux with CPython 3.14.7, Qiskit 2.4.2 and NumPy 2.5.3.
Exact installed dependency versions are recorded in `requirements-lock.txt`.

## Results

- **33 tests passed**, no failures or skips, in 44.788 seconds.
- Ruff lint and formatting checks passed for source, tests, examples and `main.py`.
- `pip check` found no dependency conflicts.
- The complete package import graph was checked and has no cycles.
- A wheel was built, installed under a separate `/tmp` directory and imported in
  isolated Python mode outside either source repository. Both general and sparse
  `m=6` circuits passed the independent DFT check; no `experiments` module was loaded.
- All three circuit-generation commands and the subcircuit example in the README
  were executed successfully. OpenQASM 3 and JSON metadata were generated.

During extraction, **42 gate-by-gate comparisons** against the original source
passed. These compared qubit count, every gate and parameter, complete physical
wire mapping and global phase: ten general circuits (five local syntheses,
terminal/coherent) and 32 sparse coherent circuits (four decoders, two sum
backends, two reductions and two cleanup modes), all at `m=6`. This one-time
migration check required the original checkout; the shipped tests do not.
Source fingerprints are in `source-fingerprints.json`.

## Coverage

- General C and A on every valid label for `(2,)`, `(3,2)`, `(3,5)`, `(4,3)` and
  `(2,3,5)`, including inverse checks on invalid physical input labels.
- General QFT at `m=6`: all DFT columns with one common phase convention, complex
  superpositions, coherent and terminal outputs, and clean workspace.
- Sparse coherent QFT with `w=0,1,2`, three original decoders and both sum backends;
  projected lookup with `w=0,1,3`, prefix reduction, both cleanup choices and both
  output modes. Every physical amplitude participates in the error comparison.
- All five explicit local-QFT syntheses compared against the `p=3` DFT.
- Arithmetic truth tables, inverse operations, count trees, carry-save sums,
  prefix adders, comparators, modular addition, reduction and rounding.
- Independent simulator checks against Qiskit, including phase and inverse gates.
- Seeded output corruption: relative-phase errors and workspace leakage both fail.
- CLI option rejection, synthesis metadata, export, existing-output protection,
  and installed module/console entry points outside the repository.

## Reproduced example

The optimized README command at `m=6,w=2` generated a native circuit with
366 qubits, 4651 gates and depth 1753. Its independent complex-state DFT check
reported L2 error `1.8994215420390052e-14`; the accumulated discarded-L2 bound was
`1.1091510935648319e-14`, with peak simulated support 3456.
These are finite-instance measurements using the exact documented options.

## Limits

Small-instance tests do not establish asymptotic resource bounds. Full circuits
use many clean ancillas, so the optional simulator tracks sparse intermediate
support and has an explicit cap. The CLI test validates the native constructed
unitary; Qiskit basis transpilation is exercised and its gate basis is checked,
but the large transpiled circuit is not independently statevector-simulated.
OpenQASM output is tested through Qiskit's serializer and output inspection;
an independent OpenQASM parser is not a dependency of this project.

The wheel installation check used the recorded Python 3.14 environment. Python
3.11+ is declared for compatible syntax and dependencies, but additional Python
versions were not available for this local validation run.
