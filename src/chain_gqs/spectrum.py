"""Classical spectrum of the fixed-N sector and helpers shared by QPE/VQE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ChainModel


@dataclass
class Spectrum:
    energies: np.ndarray        # ascending, shape (d,)
    vectors: np.ndarray         # columns are eigenvectors in sector basis, shape (d, d)
    labels: list                # site-order configuration labels of the sector basis
    sector: np.ndarray          # basis integers of the sector

    def full_vector(self, k: int, L: int) -> np.ndarray:
        """Eigenvector ``k`` embedded in the full ``2**L`` space."""
        psi = np.zeros(2**L, dtype=complex)
        psi[self.sector] = self.vectors[:, k]
        return psi

    def degenerate_with(self, k: int, atol: float = 1e-8) -> np.ndarray:
        return np.isclose(self.energies, self.energies[k], atol=atol, rtol=0)

    def table(self):
        import pandas as pd
        return pd.DataFrame({"level": range(len(self.energies)), "energy": self.energies})

    def vectors_table(self):
        import pandas as pd
        return pd.DataFrame(self.vectors, index=self.labels,
                            columns=[f"E{k}" for k in range(len(self.energies))])


def exact_spectrum(model: ChainModel) -> Spectrum:
    """Diagonalize the ``N``-electron sector Hamiltonian."""
    H2 = model.sector_hamiltonian
    E, vec = np.linalg.eigh(H2)
    return Spectrum(E, vec, model.sector_labels, model.sector)


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
