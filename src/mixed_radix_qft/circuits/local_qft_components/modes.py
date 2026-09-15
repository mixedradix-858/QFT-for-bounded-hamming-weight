"""Modes circuits and mathematical helpers."""

from __future__ import annotations

EXPLICIT_ARITHMETIC_MODES = frozenset(
    {
        "explicit",
        "explicit-fourier",
        "explicit-fourier-history",
        "explicit-register",
        "explicit-register-sklansky",
    }
)


def _uses_explicit_arithmetic(arithmetic: str) -> bool:
    """Identify arithmetic modes with fully expanded primitive circuits."""
    return arithmetic in EXPLICIT_ARITHMETIC_MODES


def _addition_backend(arithmetic: str) -> str:
    """Select the constant-addition implementation required by an MZ mode."""
    return "fourier" if "fourier" in arithmetic or "register" in arithmetic else "cuccaro"


def _uses_history_predicates(arithmetic: str) -> bool:
    """Identify the mode that retains overflow predicates for outer uncomputation."""
    return arithmetic == "explicit-fourier-history"


def _uses_register_arithmetic(arithmetic: str) -> bool:
    """Identify modes that combine controlled shifts into a register addition."""
    return arithmetic in {"explicit-register", "explicit-register-sklansky"}


def _register_adder(arithmetic: str) -> str:
    """Select Cuccaro or Sklansky for variable register arithmetic."""
    return "sklansky" if arithmetic == "explicit-register-sklansky" else "cuccaro"
