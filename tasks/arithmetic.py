"""Arithmetic reasoning tasks.

Based on Fan et al. "Length Generalization in Arithmetic Transformers" —
addition, subtraction, multiplication with configurable digit counts.
The reversed variants present digits in reverse order (LSB first), which
aligns the carry chain with left-to-right autoregressive generation and
enables better length generalization.
"""

import numpy as np


def addition(rng: np.random.Generator, level: int) -> str:
    """Addition of two numbers. Level = number of digits per operand."""
    a, b = _two_operands(rng, level)
    return f"{a}+{b}={a + b}"


def addition_rev(rng: np.random.Generator, level: int) -> str:
    """Addition with reversed digit order (LSB first)."""
    a, b = _two_operands(rng, level)
    c = a + b
    return f"{_rev(a)}+{_rev(b)}={_rev(c)}"


def subtraction(rng: np.random.Generator, level: int) -> str:
    """Subtraction (a >= b for non-negative result). Level = digits."""
    a, b = _two_operands(rng, level)
    if a < b:
        a, b = b, a
    return f"{a}-{b}={a - b}"


def multiplication(rng: np.random.Generator, level: int) -> str:
    """Multiplication. Level = number of digits per operand."""
    a, b = _two_operands(rng, level)
    return f"{a}*{b}={a * b}"


def _two_operands(rng: np.random.Generator, n_digits: int) -> tuple[int, int]:
    """Sample two n_digits-digit numbers."""
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 0
    hi = 10**n_digits
    a = int(rng.integers(lo, hi))
    b = int(rng.integers(lo, hi))
    return a, b


def _rev(n: int) -> str:
    """Reverse the digits of a non-negative integer."""
    return str(n)[::-1]
