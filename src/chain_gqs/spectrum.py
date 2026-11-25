"""Classical spectrum of the fixed-N sector and helpers shared by QPE/VQE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .basis import occupations
from .model import ChainModel


@dataclass
class Spectrum:
    energies: np.ndarray        # ascending, shape (d,)
    vectors: np.ndarray         # columns are eigenvectors in sector basis, shape (d, d)
    labels: list                # site-order configuration labels of the sector basis
    sector: np.ndarray          # basis integers of the sector
    L: Optional[int] = None     # chain length (set by exact_spectrum)

    def full_vector(self, k: int, L: int) -> np.ndarray:
        """Eigenvector ``k`` embedded in the full ``2**L`` space."""
        psi = np.zeros(2**L, dtype=complex)
        psi[self.sector] = self.vectors[:, k]
        return psi

    def degenerate_with(self, k: int, atol: float = 1e-8) -> np.ndarray:
        return np.isclose(self.energies, self.energies[k], atol=atol, rtol=0)

    def degeneracies(self, atol: float = 1e-8) -> np.ndarray:
        """Number of levels sharing each level's energy."""
        return np.array([int(self.degenerate_with(k, atol).sum()) for k in range(len(self.energies))])

    def site_densities(self) -> np.ndarray:
        """``<n_i>`` of every eigenstate, shape ``(levels, L)``."""
        occ = np.array([occupations(x, self.L) for x in self.sector])   # (d, L)
        return (np.abs(self.vectors) ** 2).T @ occ

    def table(self, decimals: int = 6):
        """Energies, excitation energies above the ground state, and degeneracies."""
        import pandas as pd
        E = self.energies
        return pd.DataFrame({"energy": np.round(E, decimals),
                             "E - E0": np.round(E - E[0], decimals),
                             "degeneracy": self.degeneracies()},
                            index=pd.Index(range(len(E)), name="level"))

    def vectors_table(self, decimals: int = 4):
        """Eigenvector amplitudes: rows are configurations, columns are levels."""
        import pandas as pd
        vals = self.vectors.real if np.allclose(self.vectors.imag, 0) else self.vectors
        df = pd.DataFrame(np.round(vals, decimals) + 0.0,   # + 0.0 turns -0.0 into 0.0
                          index=pd.Index(self.labels, name="configuration"),
                          columns=[f"E{k}" for k in range(len(self.energies))])
        return df

    def densities_table(self, decimals: int = 4):
        """Site occupations ``<n_i>`` of each eigenstate."""
        import pandas as pd
        return pd.DataFrame(np.round(self.site_densities(), decimals),
                            index=pd.Index(range(len(self.energies)), name="level"),
                            columns=[f"<n{i}>" for i in range(self.L)])


def exact_spectrum(model: ChainModel) -> Spectrum:
    """Diagonalize the ``N``-electron sector Hamiltonian."""
    H2 = model.sector_hamiltonian
    E, vec = np.linalg.eigh(H2)
    # eigh returns each eigenvector with an arbitrary sign; fix it so the
    # largest-magnitude component is positive, giving reproducible tables.
    big = np.argmax(np.abs(vec), axis=0)
    vec = vec * np.sign(vec[big, np.arange(vec.shape[1])])
    return Spectrum(E, vec, model.sector_labels, model.sector, model.L)


def spectrum_sweep(model: ChainModel, param: str, values: Sequence[float]) -> dict:
    """Diagonalize ``model.with_(**{param: v})`` for each value.

    Works for hand-built models (``param='V'``, ``'J'``, ...) and presets
    (``param='lam_ee'`` for ``townsend2021``). Returns ``values``, ``energies``
    (shape ``(len(values), d)``), ``gap`` (E1 - E0) and the ``spectra`` themselves.
    """
    values = np.asarray(values, float)
    spectra = [exact_spectrum(model.with_(**{param: float(v)})) for v in values]
    E = np.array([s.energies for s in spectra])
    return dict(param=param, values=values, energies=E, gap=E[:, 1] - E[:, 0], spectra=spectra)


def energy_bounds(model: ChainModel, margin_fraction: float = 0.05) -> tuple[float, float]:
    """Gershgorin bounds ``(a, W)`` with all sector energies in ``(a, a + W)``.

    Obtained without diagonalization; used for the QPE phase window and the
    VQD penalty strength.
    """
    A = model.sector_hamiltonian
    diag = np.diag(A)
    radius = np.sum(np.abs(A), axis=1) - np.abs(diag)
    lo, hi = np.min(diag - radius), np.max(diag + radius)
    margin = margin_fraction * max(hi - lo, 1.0)
    return float(lo - margin), float(hi - lo + 2 * margin)


def eigenspace_fidelity(sector_state: np.ndarray, spectrum: Spectrum, k: int) -> float:
    """Weight of ``sector_state`` in the (possibly degenerate) eigenspace of level ``k``."""
    V = spectrum.vectors[:, spectrum.degenerate_with(k)]
    return float(np.sum(np.abs(V.conj().T @ sector_state) ** 2))


def free_fermion_energies(L: int, N: int, J: float = 1.0) -> np.ndarray:
    """Sorted many-body energies at ``V = 0, eps = 0`` (open chain), for checks."""
    from itertools import combinations
    single = -2 * J * np.cos(np.arange(1, L + 1) * np.pi / (L + 1))
    return np.sort([sum(single[list(c)]) for c in combinations(range(L), N)])
