"""Config circuits and mathematical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, prod
from numbers import Integral
from typing import Sequence

from mixed_radix_qft.classical.crt import transformed_crt_tuple
from mixed_radix_qft.classical.number_theory import ceil_log2, is_prime, validate_moduli
from mixed_radix_qft.classical.sparse_domain import sparse_values

WEIGHTED_SUM_BACKENDS = ("wallace-qfa2", "cuccaro")


@dataclass(frozen=True)
class SparseQFTConfig:
    """Prime factors and input promise 0<=x<m, Ham(x)<=w.

    The article uses n=ceil(log2(m)); the algorithm API enforces this width."""

    n: int
    w: int
    moduli: tuple[int, ...]

    @classmethod
    def from_moduli(cls, moduli: Sequence[int], w: int, n: int | None = None) -> "SparseQFTConfig":
        """Construct a validated configuration, using canonical n unless explicitly
        supplied."""
        normalized = tuple(moduli)
        validate_moduli(normalized)
        canonical_n = ceil_log2(prod(normalized))
        config = cls(n=canonical_n if n is None else n, w=w, moduli=normalized)
        config.validate()
        return config

    @property
    def crt_modulus(self) -> int:
        """Return m=product(moduli), the logical transform order."""
        return prod(self.moduli)

    @property
    def canonical_n(self) -> int:
        """Return ceil(log2(m)), the canonical global register width."""
        return ceil_log2(self.crt_modulus)

    @property
    def widths(self) -> tuple[int, ...]:
        """Return the physical widths ceil(log2(m_j)) of the little-endian fields."""
        return tuple((ceil_log2(modulus) for modulus in self.moduli))

    @property
    def q_y(self) -> int:
        """Return the sum of local field widths, including unused physical labels."""
        return sum(self.widths)

    @property
    def sparse_domain(self) -> tuple[int, ...]:
        """Enumerate x<min(m,2**n) with Ham(x)<=w."""
        return sparse_values(self.n, self.w, upper_bound=self.crt_modulus)

    @property
    def transformed_labels(self) -> tuple[tuple[int, ...], ...]:
        """Enumerate eta(x)=(g_j*x mod m_j)_j over the promised inputs."""
        return tuple((transformed_crt_tuple(value, self.moduli) for value in self.sparse_domain))

    def validate(self) -> None:
        """Check dimensions and distinct prime moduli without enumerating the sparse
        domain."""
        if any((not isinstance(v, Integral) or isinstance(v, bool) for v in (self.n, self.w))):
            raise ValueError("n and w must be integers")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.w < 0 or self.w > self.n:
            raise ValueError("w must satisfy 0 <= w <= n")
        if not self.moduli:
            raise ValueError("at least one modulus is required")
        if len(set(self.moduli)) != len(self.moduli):
            raise ValueError("the construction requires distinct moduli")
        if any((not is_prime(modulus) for modulus in self.moduli)):
            raise ValueError("the construction construction requires prime moduli")
        for index, left in enumerate(self.moduli):
            for right in self.moduli[index + 1 :]:
                if gcd(left, right) != 1:
                    raise ValueError("moduli must be pairwise coprime")
