"""Spinless extended-Hubbard (Coulomb) chain.

.. math::

    H = -J \\sum_{i=0}^{L-2} (c_i^\\dagger c_{i+1} + h.c.)
        + \\sum_{i<j} V_{ij} n_i n_j + \\sum_i \\epsilon_i n_i,

open boundary conditions, hbar = 1. By default ``V_ij = V / |i - j|``
(unit lattice spacing). Supply ``Vij`` to override.

This module is pure NumPy; :meth:`ChainModel.sparse_pauli_op` imports Qiskit lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Optional

import numpy as np

from .basis import number_operator_diagonal, occupations, sector_indices, site_label


def coulomb_matrix(L: int, V: float, power: float = 1.0) -> np.ndarray:
    """Symmetric ``L x L`` matrix ``V / |i - j|**power`` with zero diagonal."""
    Vij = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            Vij[i, j] = Vij[j, i] = V / (j - i) ** power
    return Vij


def nearest_neighbor_matrix(L: int, V: float) -> np.ndarray:
    """Interaction only between neighbouring sites (``V_{i,i+1} = V``)."""
    Vij = np.zeros((L, L))
    for i in range(L - 1):
        Vij[i, i + 1] = Vij[i + 1, i] = V
    return Vij


@dataclass(frozen=True)
class ChainModel:
    """Parameters of the chain. Immutable; use :meth:`with_` for sweeps.

    Parameters
    ----------
    L : number of sites.
    N : number of electrons (used for sector restriction and GQS checks).
    J : hopping amplitude.
    V : Coulomb strength for the default ``V/|i-j|`` interaction.
    eps : site energies, length ``L`` (default zeros).
    Vij : explicit interaction matrix; replaces the default when given.
    preset, params : set by the factories in :mod:`chain_gqs.presets`. They record
        which physical parameters produced ``Vij``/``eps`` so that :meth:`with_`
        can rebuild the model, and :attr:`label` can describe it. Leave them
        empty when constructing a model by hand.
    """

    L: int
    N: int = 2
    J: float = 1.0
    V: float = 1.0
    eps: Optional[tuple] = None
    Vij: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    preset: str = ""
    params: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if self.L < 2 or not 0 <= self.N <= self.L:
            raise ValueError("Require L >= 2 and 0 <= N <= L")
        if not np.all(np.isfinite([self.J, self.V])):
            raise ValueError("J and V must be finite")
        eps = np.zeros(self.L) if self.eps is None else np.asarray(self.eps, float)
        if eps.shape != (self.L,) or not np.all(np.isfinite(eps)):
            raise ValueError("eps must contain L finite site energies")
        object.__setattr__(self, "eps", tuple(eps))
        if self.Vij is not None:
            M = np.asarray(self.Vij, float)
            if (M.shape != (self.L, self.L) or not np.all(np.isfinite(M))
                    or not np.allclose(M, M.T) or not np.allclose(M.diagonal(), 0)):
                raise ValueError("Vij must be finite, symmetric L x L with zero diagonal")
            object.__setattr__(self, "Vij", M)

    # ------------------------------------------------------------------ basics
    def with_(self, **changes) -> "ChainModel":
        """Copy with some parameters changed.

        * Hand-built model: ``model.with_(V=2.0)``. Raises if you change ``V``
          while an explicit ``Vij`` is set, since ``V`` would then have no effect.
        * Preset model: change the preset's own parameters, e.g.
          ``model.with_(lam_ee=5.0)``; ``Vij`` and ``eps`` are rebuilt.
        """
        if self.preset:
            builder = _PRESET_BUILDERS[self.preset]
            unknown = set(changes) - set(self.params)
            if unknown:
                raise ValueError(
                    f"model preset '{self.preset}' is defined by {sorted(self.params)}; "
                    f"cannot change {sorted(unknown)} directly")
            return builder(**{**self.params, **changes})
        if self.Vij is not None and "V" in changes and "Vij" not in changes:
            raise ValueError("V has no effect when an explicit Vij is set; pass a new Vij instead")
        return replace(self, **changes)

    @property
    def label(self) -> str:
        """Short description for plot titles and saved results."""
        if self.preset:
            def fmt(v):
                if isinstance(v, bool):
                    return str(v)
                if isinstance(v, (int, float, np.number)):
                    return f"{v:g}"
                return "custom" if isinstance(v, (tuple, list, np.ndarray)) else str(v)
            args = ", ".join(f"{k}={fmt(v)}" for k, v in self.params.items() if v is not None)
            return f"{self.preset}({args})"
        return f"L={self.L}, N={self.N}, J={self.J:g}, V={self.V:g}"

    @property
    def site_energies(self) -> np.ndarray:
        return np.asarray(self.eps, float)

    @property
    def interactions(self) -> np.ndarray:
        """The interaction matrix actually used."""
        return coulomb_matrix(self.L, self.V) if self.Vij is None else np.asarray(self.Vij)

    @property
    def dim(self) -> int:
        return 2**self.L

    @cached_property
    def sector(self) -> np.ndarray:
        """Basis integers of the ``N``-electron sector."""
        return sector_indices(self.L, self.N)

    @property
    def sector_labels(self) -> list[str]:
        return [site_label(x, self.L) for x in self.sector]

    # ------------------------------------------------------------ Hamiltonian
    @cached_property
    def hamiltonian(self) -> np.ndarray:
        """Dense ``2**L x 2**L`` Hamiltonian on the full qubit space."""
        L, J = self.L, self.J
        eps, Vij = self.site_energies, self.interactions
        H = np.zeros((2**L, 2**L))
        iu = np.triu_indices(L, 1)
        for x in range(2**L):
            n = occupations(x, L)
            H[x, x] = eps @ n + np.sum(Vij[iu] * n[iu[0]] * n[iu[1]])
            for i in range(L - 1):
                if n[i] != n[i + 1]:
                    # Adjacent hop: no Jordan-Wigner sign between neighbours.
                    H[x ^ (1 << i) ^ (1 << (i + 1)), x] = -J
        return H

    @property
    def sector_hamiltonian(self) -> np.ndarray:
        """Hamiltonian restricted to the ``N``-electron sector."""
        idx = self.sector
        return self.hamiltonian[np.ix_(idx, idx)]

    def pauli_terms(self) -> list[tuple[str, float]]:
        """Jordan-Wigner Pauli decomposition as ``(label, coeff)`` pairs.

        Labels follow Qiskit's convention (rightmost character = qubit 0), so the
        list can be passed straight to ``SparsePauliOp.from_list``. Terms are not
        simplified; :func:`combine_pauli_terms` merges duplicates.
        """
        L, J = self.L, self.J
        eps, Vij = self.site_energies, self.interactions
        terms: list[tuple[str, float]] = []

        def add(c, factors):
            label = ["I"] * L
            for i, p in factors.items():
                label[L - 1 - i] = p
            terms.append(("".join(label), float(c)))

        for i in range(L - 1):
            add(-J / 2, {i: "X", i + 1: "X"})
            add(-J / 2, {i: "Y", i + 1: "Y"})
        for i in range(L):
            add(eps[i] / 2, {})
            add(-eps[i] / 2, {i: "Z"})
            for j in range(i + 1, L):
                w = Vij[i, j] / 4
                add(w, {}); add(-w, {i: "Z"}); add(-w, {j: "Z"}); add(w, {i: "Z", j: "Z"})
        return combine_pauli_terms(terms)

    def sparse_pauli_op(self):
        """Qiskit ``SparsePauliOp`` (requires Qiskit)."""
        from qiskit.quantum_info import SparsePauliOp
        return SparsePauliOp.from_list(self.pauli_terms()).simplify()

    def conserves_number(self, atol: float = 1e-12) -> bool:
        n = number_operator_diagonal(self.L)
        return bool(np.allclose(self.hamiltonian * (n[:, None] - n[None, :]), 0, atol=atol))


# Registry filled by chain_gqs.presets; lets ChainModel.with_ rebuild preset models.
_PRESET_BUILDERS: dict = {}


def register_preset(name: str):
    """Decorator: register a factory returning a ChainModel with ``preset=name``."""
    def deco(fn):
        _PRESET_BUILDERS[name] = fn
        return fn
    return deco


def combine_pauli_terms(terms, atol: float = 1e-14) -> list[tuple[str, float]]:
    """Merge duplicate Pauli labels and drop zero coefficients."""
    out: dict[str, float] = {}
    for label, c in terms:
        out[label] = out.get(label, 0.0) + c
    return [(k, v) for k, v in out.items() if abs(v) > atol]


_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Qiskit-ordered Pauli label (leftmost = highest qubit)."""
    M = np.array([[1.0]])
    for p in label:
        M = np.kron(M, _PAULI[p])
    return M


def pauli_terms_matrix(terms) -> np.ndarray:
    """Dense matrix of a list of ``(label, coeff)`` pairs."""
    return sum(c * pauli_matrix(lbl) for lbl, c in terms)
