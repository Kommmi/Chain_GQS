"""Named parameter sets for :class:`~chain_gqs.model.ChainModel`.

Each factory converts physical parameters into the ``J``, ``Vij`` and ``eps``
that the rest of the package uses, and records those parameters on the model so
that ``model.with_(...)`` rebuilds it consistently during a sweep:

>>> m = townsend2021(L=5, N=2, lam_ee=1.0)
>>> m.with_(lam_ee=10.0)     # Vij AND eps are rescaled together

Available presets
-----------------
``coulomb``      -- ``V / |i - j|`` with optional site energies (the original notebooks).
``townsend2021`` -- Townsend et al., arXiv:2011.07113, Eqs. (1)-(3).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .model import ChainModel, coulomb_matrix, register_preset


# --------------------------------------------------------------- building blocks
def screened_coulomb_matrix(L: int, lam_ee: float, f_ex: float = 0.2, zeta: float = 0.5) -> np.ndarray:
    """``V_ij = lam_ee (1 - f_ex delta_{|i-j|,1}) / (|i - j| + zeta)``, zero diagonal.

    Townsend et al. Eq. (2): Coulomb repulsion with an orbital-size cutoff
    ``zeta`` and nearest-neighbour exchange reduction ``f_ex``.
    """
    d = np.abs(np.subtract.outer(np.arange(L), np.arange(L))).astype(float)
    V = lam_ee * (1 - f_ex * (d == 1)) / (d + zeta)
    np.fill_diagonal(V, 0.0)
    return V


def core_potential(L: int, lam_nuc: float, Z: float, zeta: float = 0.5,
                   include_self: bool = True) -> np.ndarray:
    """``eps_i = -lam_nuc Z sum_j 1 / (|i - j| + zeta)`` (Townsend et al. Eq. 1).

    ``include_self=True`` keeps the electron's own core (``j = i``, giving
    ``1/zeta``); the paper sums over "each of the atoms", which we read as
    including it. It only shifts all sites equally, so it does not affect
    dynamics at fixed electron number.
    """
    d = np.abs(np.subtract.outer(np.arange(L), np.arange(L))).astype(float)
    terms = 1.0 / (d + zeta)
    if not include_self:
        np.fill_diagonal(terms, 0.0)
    return -lam_nuc * Z * terms.sum(axis=1)


# --------------------------------------------------------------------- presets
@register_preset("coulomb")
def coulomb(L: int, N: int = 2, J: float = 1.0, V: float = 1.0,
            eps: Optional[Sequence[float]] = None) -> ChainModel:
    """The original notebook model: ``V/|i-j|`` repulsion, hopping ``J``, site energies ``eps``."""
    eps_t = None if eps is None else tuple(float(e) for e in eps)
    return ChainModel(L=L, N=N, J=J, V=V, eps=eps_t, preset="coulomb",
                      params=dict(L=L, N=N, J=J, V=V, eps=eps_t))


@register_preset("townsend2021")
def townsend2021(L: int, N: int = 2, lam_ee: float = 1.0, t: float = 1.0,
                 f_ex: float = 0.2, zeta_ee: float = 0.5, zeta_nuc: float = 0.5,
                 lam_nuc: Optional[float] = None, Z: Optional[float] = None,
                 include_self: bool = True) -> ChainModel:
    """Townsend, Neuman, Debrecht, Aizpurua & Bryant, arXiv:2011.07113.

    Parameters (paper's defaults)
    -----------------------------
    lam_ee : electron-electron Coulomb strength; the sweep parameter is ``lam_ee / t``.
    t : nearest-neighbour hopping (``J`` in this package).
    f_ex : nearest-neighbour exchange reduction, 0.2 (SI also uses 0 and 0.3).
    zeta_ee, zeta_nuc : orbital cutoffs, 0.5 lattice spacings.
    lam_nuc : core-attraction strength. ``None`` means ``lam_nuc = lam_ee``:
        the paper says the two scale together but does not state the ratio.
    Z : nuclear charge per site. ``None`` means ``N / L`` (charge neutral).
    include_self : include the electron's own core in the potential sum.
    """
    lam_nuc_eff = lam_ee if lam_nuc is None else lam_nuc
    Z_eff = N / L if Z is None else Z
    Vij = screened_coulomb_matrix(L, lam_ee, f_ex, zeta_ee)
    eps = core_potential(L, lam_nuc_eff, Z_eff, zeta_nuc, include_self)
    params = dict(L=L, N=N, lam_ee=lam_ee, t=t, f_ex=f_ex, zeta_ee=zeta_ee,
                  zeta_nuc=zeta_nuc, lam_nuc=lam_nuc, Z=Z, include_self=include_self)
    return ChainModel(L=L, N=N, J=t, V=lam_ee, eps=tuple(eps), Vij=Vij,
                      preset="townsend2021", params=params)


PRESETS = {"coulomb": coulomb, "townsend2021": townsend2021}

__all__ = ["screened_coulomb_matrix", "core_potential", "coulomb", "townsend2021",
           "PRESETS", "coulomb_matrix"]
