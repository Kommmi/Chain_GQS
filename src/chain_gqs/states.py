"""Initial-state constructors (full ``2**L`` statevectors, NumPy only)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .basis import index_from_label, index_from_occupied, sector_indices


def product_state(L: int, occupied: Sequence[int]) -> np.ndarray:
    """Occupation-basis state with the listed sites filled."""
    psi = np.zeros(2**L, dtype=complex)
    psi[index_from_occupied(occupied, L)] = 1.0
    return psi


def state_from_label(label: str) -> np.ndarray:
    """Occupation-basis state from a site-order label such as ``'1010'``."""
    psi = np.zeros(2 ** len(label), dtype=complex)
    psi[index_from_label(label)] = 1.0
    return psi


def random_sector_state(L: int, N: int, seed=None, real: bool = True) -> np.ndarray:
    """Normalized random state supported only on the ``N``-electron sector."""
    rng = np.random.default_rng(seed)
    idx = sector_indices(L, N)
    amp = rng.normal(size=len(idx))
    if not real:
        amp = amp + 1j * rng.normal(size=len(idx))
    psi = np.zeros(2**L, dtype=complex)
    psi[idx] = amp / np.linalg.norm(amp)
    return psi


def as_statevector_array(state) -> np.ndarray:
    """Accept a NumPy array or a Qiskit ``Statevector`` and return a flat array."""
    data = getattr(state, "data", state)
    return np.asarray(data, dtype=complex).reshape(-1)
