# QFT for bounded hamming weight

Explicit Python/Qiskit circuits for the article **Low-Depth Quantum Fourier
Transform Circuits for Bounded-Hamming-Weight Inputs**.

The package implements both the general mixed-radix QFT and its bounded-Hamming-weight
input construction. Circuit generation, optional simulation and basis transpilation
are separate operations.

## Install

Requires Python 3.11 or later. Qiskit is pinned to the source implementation's version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For use without development tools, install with `python -m pip install .`.

## Generate a circuit

### General QFT

```bash
python main.py general --moduli 2 3 \
  --local-synthesis explicit-register-sklansky \
  --mode coherent --output outputs/general
```

### Sparse QFT using the article's input decoder and reduction

```bash
python main.py sparse --moduli 2 3 --w 1 \
  --local-synthesis explicit-register-sklansky \
  --weighted-sum wallace-qfa2 --decoder lookup \
  --reduction cuccaro --cleanup local \
  --mode coherent --output outputs/sparse
```

### Sparse QFT using the optimized implementation variants

```bash
python main.py sparse --moduli 2 3 --w 2 \
  --local-synthesis explicit-register-sklansky \
  --weighted-sum wallace-qfa2 --decoder lookup-projected \
  --reduction prefix --cleanup deferred \
  --mode coherent --basis u-cx --optimization-level 1 \
  --simulate --output outputs/sparse-optimized
```

Each output directory must be new. It receives `circuit.qasm` (OpenQASM 3) and
`metadata.json` with the actual synthesis, register mapping, versions and resource
counts. Without `--output`, the circuit is still built and its report is printed.
`--draw` prints the generated circuit as text. Run `python main.py --help` for all options.

`python -m mixed_radix_qft` and the installed `mixed-radix-qft` command expose the
same interface, including when invoked outside this checkout.

## Mathematics and register conventions

Let `m=product(moduli)` and `n=ceil(log2(m))`. Factors are supplied in register order.
The full algorithms require distinct **prime** factors; lower-level `C` and `A`
also support pairwise coprime composite factors. The prime-order local QFT backend
does not extend that support to complete composite-factor circuits.

The Fourier convention is

```text
F_m[y,x] = exp(+2*pi*i*x*y/m) / sqrt(m).
```

All integer fields are little-endian. The input occupies the first `n` qubits,
followed by the residue fields and workspace. Every register other than the input
starts in zero.

| Block | Action on the valid input domain |
| --- | --- |
| `C` | `|x>|0> -> |0>|chi(x)>`, with `chi_j(x)=x mod m_j` |
| `A` | `|r_j> -> |g_j*r_j mod m_j>`, with `g_j=(m/m_j)^(-1) mod m_j` |
| Local `F_mj` | Positive-phase QFT on each prime-dimensional field |
| General QFT | `C† (tensor_j F_mj) A C` |
| Sparse conversion | `|x>|0> -> |0>|eta(x)>`, with `eta_j(x)=g_j*x mod m_j` |
| Sparse QFT | `(general C†) (tensor_j F_mj) C_sparse^(A)` |

General inputs satisfy `0<=x<m`. Sparse inputs additionally satisfy `Ham(x)<=w`,
with `0<=w<=n`. These promises also apply to the support of a superposition.
The sparse promise constrains the **input**; the Fourier output can be dense.
The final coherent `C†` therefore always uses the general CRT conversion inverse.

- **`--mode coherent`** returns `F_m|psi>` on the input register, with every
  residue and workspace qubit restored to zero.
- **`--mode terminal`** omits `C†` and returns Fourier amplitudes in the ordinary
  CRT residue fields. Measurement outcomes need classical CRT reconstruction;
  this is not a coherent output on the binary input register.

Use `classical.crt.decode_fields` and `classical.crt.crt_inverse` to reconstruct
terminal measurement outcomes. The terminal fields encode `chi(y)`, not `eta(y)`.

Arithmetic promises apply on valid logical labels. The gates define a reversible
physical extension outside that domain, but no identity or clean-work guarantee
is claimed there.

## Synthesis choices

The defaults are `explicit-register-sklansky` for the local QFT and, for sparse
conversion, `wallace-qfa2`, `lookup`, `cuccaro`, `local`.

| Option | Supported values and meaning |
| --- | --- |
| `--local-synthesis` | `explicit`: controlled constant additions using Cuccaro; `explicit-fourier`: shared Fourier additions; `explicit-fourier-history`: retain overflow history for outer uncomputation; `explicit-register`: whole-register Cuccaro arithmetic; `explicit-register-sklansky`: whole-register Sklansky arithmetic |
| `--weighted-sum` | `wallace-qfa2`: carry-save tree with final Sklansky sum; `cuccaro`: balanced ripple-adder reference tree |
| `--decoder` | `lookup`: full promised-domain table; `lookup-projected`: certified injective projection of that table's keys; `crt-inverse`: arithmetic decoder using Sklansky sums; `crt-kogge-stone`: arithmetic decoder using Kogge–Stone sum trees; `crt-wallace-kogge-stone`: carry-save CRT decoder with final Kogge–Stone additions |
| `--reduction` | `cuccaro`: threshold comparisons and controlled subtractions; `prefix`: threshold flags, carry-save correction and prefix sum |
| `--cleanup` | `local`: clear internal reduction work immediately; `deferred`: retain it until the surrounding compute-copy-uncompute clears it |
| `--lookup-entry-limit` | Maximum enumerated input-decoder table size; default 100000 for both lookup decoders |

The optional `crt-wallace-kogge-stone` decoder replaces both CRT sum trees
with Wallace/QFA2 compression followed by a final Kogge–Stone addition.
It preserves coherent input cleanup and has O(log n) logical depth with
sufficient all-to-all workspace. See [derivation, tests and measurements](docs/carry_save_decoder.md).

The general `C` and `A` use their existing explicit carry-save/Kogge–Stone
implementation. Sparse-only options are rejected for the general algorithm.
Counter trees use Cuccaro adders; they do not inherit the local-QFT adder choice.

The main article describes `T_j`, contribution lookup `L_j^(A)`, carry-save `W_j`,
threshold/controlled-subtraction `R_j`, and the full lookup input decoder.
Projected keys, arithmetic input decoders, prefix reduction and the alternative
local arithmetic modes are implementation variants retained for explicit selection.
The coherent completion is supplied by the general mixed-radix identity; the
article's terminal measurement path remains available separately.

## Transpilation and resource counts

`--basis native` retains the emitted explicit gates. Native depth counts
`X`, `CX`, `CCX`, `H`, `P`, `CP`, `RY`, `CRY`, `SWAP` and `Z` as individual gates.


`--basis u-cx` or `--basis rz-sx-x-cx` additionally invokes Qiskit transpilation.
The original and transpiled counts occupy separate report fields. Compilation
uses all-to-all connectivity, `--optimization-level` (default 1 when transpiling),
and `--seed` (default 7). No hardware routing or finite fault-tolerant rotation
synthesis is implied.


## Python API

```python
from mixed_radix_qft import SynthesisOptions, build_qft

circuit = build_qft(
    "sparse", (2, 3), w=1,
    synthesis=SynthesisOptions(
        local_backend="explicit-register-sklansky",
        decoder="lookup",
        weighted_sum_backend="wallace-qfa2",
        reduction_backend="cuccaro",
        reduction_cleanup="local",
    ),
)
```

Individual builders return `QuantumCircuit` objects or documented circuit/layout
pairs. Examples of `C`, `A`, local QFT and counter construction are in
[`examples/build_subcircuits.py`](examples/build_subcircuits.py).

## Layout

```text
main.py                         Single command-line entry point
src/mixed_radix_qft/
  algorithms/general.py         General composition
  algorithms/sparse.py          Sparse composition and general final C†
  circuits/c.py                 General CRT conversion
  circuits/a.py                 Good–Thomas correction
  circuits/local_qft.py          Local QFT assembly
  circuits/local_qft_components/ Range preparation, QFS, estimator,
                                reflections, amplification, QFP and arithmetic
  circuits/counters.py          T_j
  circuits/contributions.py     L_j^(A)
  circuits/weighted_sum.py      W_j
  circuits/modular_reduction.py R_j
  circuits/residues.py          U_j and C_res^(A)
  circuits/sparse_conversion.py Input cleanup and fused A C
  circuits/adders/              Cuccaro, Sklansky, Kogge–Stone, QFA2 and constants
  classical/                   CRT, input domains and decoder-table preparation
  validation/                  Independent small DFT and bounded-support simulator
  synthesis.py                 Construction selection and basis transpilation
  export.py                    OpenQASM serialization
  cli.py                       Argument handling and execution
```


## Validation

```bash
python -m unittest discover -s tests -v
ruff check src tests examples main.py
ruff format --check src tests examples main.py
```

`--simulate` verifies one seeded complex input against an independently computed
DFT, including relative phase and every workspace wire. It is restricted to `m<=32`
and uses a bounded-support simulator, not a dense `2**num_qubits` vector.
`--max-states` bounds its support; exceeding the cap fails explicitly.
Simulation is optional and never runs during ordinary circuit generation.


For the exact Linux/CPython environment used during extraction, see
`requirements-lock.txt`. Install it before `python -m pip install --no-deps .`;
other Python versions may require compatible dependency resolutions instead.
