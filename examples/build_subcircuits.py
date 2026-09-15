"""Build individual article blocks using the installed package."""

from mixed_radix_qft.circuits.a import build_general_a
from mixed_radix_qft.circuits.c import build_general_c
from mixed_radix_qft.circuits.counters import build_counter_forest_circuit
from mixed_radix_qft.circuits.local_qft import build_mosca_zalka_inplace_qft


def main() -> None:
    blocks = {
        "C": build_general_c((2, 3)),
        "A": build_general_a((2, 3)),
        "F_3": build_mosca_zalka_inplace_qft(3, arithmetic="explicit-register-sklansky"),
        "T_3": build_counter_forest_circuit(3, 3, 2, explicit_primitives=True),
    }
    for name, block in blocks.items():
        circuit = block[0] if isinstance(block, tuple) else block
        print(
            f"{name}: {circuit.num_qubits} qubits, {len(circuit.data)} gates, depth {circuit.depth()}"
        )


if __name__ == "__main__":
    main()
