"""Bit-ordering conventions and particle-number sectors.

Conventions used throughout the package
---------------------------------------
* Site ``i`` is qubit ``i``.
* A basis state is an integer ``x = sum_i n_i * 2**i`` (Qiskit's little-endian
  statevector index).
* Human-readable labels are always written in **increasing site order**,
  ``|n_0 n_1 ... n_{L-1}>``. Qiskit prints the reverse, ``|q_{L-1} ... q_0>``.
* For a list of sites ``(s_0, ..., s_{k-1})`` (a subsystem or an environment),
  the *local index* puts the first listed site in the most significant bit:
  ``a = sum_j n_{s_j} * 2**(k-1-j)``, so ``format(a, f'0{k}b')`` reads in the
  listed site order. For subsystem ``(0, 1)`` this gives the ordering
  ``[|00>, |01>, |10>, |11>]`` used in the original notebooks.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def popcount(x: int) -> int:
    """Number of set bits (occupied sites) in ``x``."""
    return int(x).bit_count()


def occupations(x: int, L: int) -> np.ndarray:
    """Occupation vector ``[n_0, ..., n_{L-1}]`` of basis integer ``x``."""
    return np.array([(int(x) >> i) & 1 for i in range(L)], dtype=int)


def site_label(x: int, L: int) -> str:
    """Site-order label ``'n_0 n_1 ... n_{L-1}'`` of basis integer ``x``."""
    return "".join(str((int(x) >> i) & 1) for i in range(L))


def index_from_label(label: str) -> int:
    """Inverse of :func:`site_label`."""
    if set(label) - {"0", "1"}:
        raise ValueError("label must contain only 0 and 1")
    return sum(1 << i for i, b in enumerate(label) if b == "1")


def index_from_occupied(occupied: Sequence[int], L: int) -> int:
    """Basis integer with the listed sites occupied."""
    occupied = tuple(int(i) for i in occupied)
    if len(set(occupied)) != len(occupied) or any(not 0 <= i < L for i in occupied):
        raise ValueError(f"occupied must list distinct sites in 0..{L - 1}")
    return sum(1 << i for i in occupied)


def sector_indices(L: int, N: int) -> np.ndarray:
    """Basis integers with exactly ``N`` electrons, sorted by site combinations."""
    return np.array([sum(1 << i for i in c) for c in combinations(range(L), N)], dtype=int)


def number_operator_diagonal(L: int) -> np.ndarray:
    """Total electron number of every basis integer ``0 .. 2**L - 1``."""
    return np.array([popcount(x) for x in range(2**L)], dtype=int)


def local_index(x: int, sites: Sequence[int]) -> int:
    """Local index of ``x`` restricted to ``sites`` (first site = most significant)."""
    k = len(sites)
    return sum(((int(x) >> s) & 1) << (k - 1 - j) for j, s in enumerate(sites))


def complement(sites: Sequence[int], L: int) -> tuple[int, ...]:
    """Sites of the chain not in ``sites``, in increasing order."""
    s = set(sites)
    return tuple(i for i in range(L) if i not in s)


def split_amplitudes(psi: np.ndarray, L: int, subsystem: Sequence[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape a full statevector into ``A[s, a]`` (subsystem index, environment index).

    Returns ``(A, environment_sites)``. Both indices use :func:`local_index`.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    if psi.size != 2**L:
        raise ValueError(f"statevector must have length 2**L = {2**L}")
    subsystem = tuple(int(s) for s in subsystem)
    if len(set(subsystem)) != len(subsystem) or any(not 0 <= s < L for s in subsystem):
        raise ValueError(f"subsystem must list distinct sites in 0..{L - 1}")
    env = complement(subsystem, L)
    A = np.zeros((2 ** len(subsystem), 2 ** len(env)), dtype=complex)
    for x, amp in enumerate(psi):
        A[local_index(x, subsystem), local_index(x, env)] = amp
    return A, env
