"""Projected ensemble / geometric quantum state (GQS) of a subsystem.

Write the global state as ``|Psi> = sum_{s,a} A[s,a] |s>_S |a>_E``. Conditioning
on environment occupation pattern ``a`` gives

    lambda_a = sum_s |A[s,a]|**2,   |chi_a> = A[:, a] / sqrt(lambda_a).

For a **two-site** subsystem, the one-electron sector ``{|01>, |10>}`` is an
encoded qubit and each ``|chi_a>`` in it gets Bloch angles

    |chi> = cos(theta/2)|01> + e^{i phi} sin(theta/2)|10>   (up to global phase),

so ``theta = 0`` is ``|01>`` (electron on the second listed site) and
``theta = pi`` is ``|10>``.

Everything here works for any chain length ``L``, electron number ``N`` and
subsystem sites; nothing depends on Qiskit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .basis import popcount, site_label, split_amplitudes
from .states import as_statevector_array


# ---------------------------------------------------------------- Bloch helpers
def bloch_angles(c01: complex, c10: complex, tol: float = 1e-12) -> tuple[float, float, bool]:
    """``(theta, phi, phi_defined)`` of ``c01|01> + c10|10>`` (need not be normalized).

    At a pole ``phi`` is undefined; it is returned as 0 with ``phi_defined=False``.
    """
    theta = float(2 * np.arctan2(abs(c10), abs(c01)))
    defined = bool(abs(c01) > tol and abs(c10) > tol)
    phi = float(np.angle(c10 * np.conj(c01)) % (2 * np.pi)) if defined else 0.0
    return theta, phi, defined


def bloch_vector(theta, phi) -> np.ndarray:
    """Cartesian ``(x, y, z)`` on the unit sphere; works on arrays."""
    theta, phi = np.asarray(theta), np.asarray(phi)
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def two_site_state_from_angles(theta: float, phi: float) -> np.ndarray:
    """Two-site vector in order ``[|00>, |01>, |10>, |11>]``."""
    return np.array([0, np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2), 0])


# ------------------------------------------------------------- reduced states
def reduced_density_matrix(psi, L: int, subsystem: Sequence[int] = (0, 1)) -> np.ndarray:
    """``rho_S`` in the local subsystem ordering of :mod:`chain_gqs.basis`."""
    A, _ = split_amplitudes(as_statevector_array(psi), L, subsystem)
    return A @ A.conj().T


def number_distribution(psi, L: int) -> np.ndarray:
    """``P(N = n)`` for ``n = 0..L``."""
    p = np.abs(as_statevector_array(psi)) ** 2
    out = np.zeros(L + 1)
    for x, w in enumerate(p):
        out[popcount(x)] += w
    return out


# ------------------------------------------------------------------- ensemble
@dataclass
class ProjectedEnsemble:
    """Result of :func:`projected_ensemble`.

    Supports ``ens['theta']``-style access so code written for the original
    dictionary output keeps working.

    Attributes
    ----------
    ensemble : list of dicts, one per retained environment outcome, with keys
        ``env, weight, n_system, chi`` and, for two-site subsystems,
        ``theta, phi, phi_defined`` (NaN angles outside the one-electron sector).
    theta, phi, weights, environment : arrays over the one-electron members only.
        ``weights`` are **unconditional** (sum to ``P(N_S = 1)``).
    sector_weights : ``{n: P(N_S = n)}``.
    discarded_weight : total weight of outcomes with ``lambda <= tol``.
    """

    L: int
    N: int
    subsystem: tuple
    environment_sites: tuple
    ensemble: list
    sector_weights: dict
    discarded_weight: float
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    environment: list = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def members(self, n_system: Optional[int] = None) -> list[dict]:
        return [r for r in self.ensemble if n_system is None or r["n_system"] == n_system]

    def density_matrix(self) -> np.ndarray:
        """``sum_a lambda_a |chi_a><chi_a|`` -- equals the reduced density matrix."""
        k = len(self.subsystem)
        rho = np.zeros((2**k, 2**k), dtype=complex)
        for r in self.ensemble:
            rho += r["weight"] * np.outer(r["chi"], r["chi"].conj())
        return rho

    def bloch_vectors(self) -> np.ndarray:
        """``(M, 3)`` Cartesian points of the one-electron members."""
        return bloch_vector(self.theta, self.phi)

    def sector_probabilities(self) -> np.ndarray:
        """``[P(N_S=0), P(N_S=1), ...]`` as an array (simplex coordinates)."""
        return np.array([self.sector_weights[n] for n in sorted(self.sector_weights)])

    def table(self):
        """Rows as a ``pandas.DataFrame`` (requires pandas)."""
        import pandas as pd
        cols = [c for c in ("env", "weight", "n_system", "theta", "phi", "phi_defined")
                if c in (self.ensemble[0] if self.ensemble else {})]
        return pd.DataFrame([{c: r[c] for c in cols} for r in self.ensemble])


def _infer_N(psi: np.ndarray, tol: float) -> int:
    weights = {}
    for x, a in enumerate(psi):
        w = abs(a) ** 2
        if w > tol:
            weights[popcount(x)] = weights.get(popcount(x), 0.0) + w
    if len(weights) != 1:
        raise ValueError("state does not have a definite electron number; pass N explicitly")
    return next(iter(weights))


def projected_ensemble(state, L: int, subsystem: Sequence[int] = (0, 1),
                       N: Optional[int] = None, tol: float = 1e-12) -> ProjectedEnsemble:
    """Projected ensemble of ``subsystem`` conditioned on environment occupations.

    Parameters
    ----------
    state : full ``2**L`` statevector (NumPy array or Qiskit ``Statevector``).
    L : number of sites.
    subsystem : sites forming the subsystem, in the order used for labels.
    N : total electron number. If ``None`` it is inferred; the state is rejected
        if it has more than ``tol`` probability outside the ``N`` sector.
    tol : outcomes with ``lambda <= tol`` are discarded (their weight is reported).
    """
    if not 0 < tol < 1:
        raise ValueError("tol must lie between 0 and 1")
    psi = as_statevector_array(state)
    if psi.size != 2**L or not np.isclose(np.vdot(psi, psi).real, 1):
        raise ValueError(f"state must be a normalized {L}-qubit statevector")
    if N is None:
        N = _infer_N(psi, tol)
    leakage = sum(abs(a) ** 2 for x, a in enumerate(psi) if popcount(x) != N)
    if leakage > tol:
        raise ValueError(f"state has probability {leakage:.3g} outside the N={N} sector")

    subsystem = tuple(int(s) for s in subsystem)
    A, env_sites = split_amplitudes(psi, L, subsystem)
    k, ke = len(subsystem), len(env_sites)
    two_site = k == 2

    rows, discarded = [], 0.0
    sector = {n: 0.0 for n in range(k + 1)}
    for a in range(2**ke):
        weight = float(np.vdot(A[:, a], A[:, a]).real)
        if weight <= tol:
            discarded += weight
            continue
        ns = N - popcount(a)
        chi = A[:, a] / np.sqrt(weight)
        row = dict(env=format(a, f"0{ke}b") if ke else "", weight=weight, n_system=ns, chi=chi)
        if two_site:
            if ns == 1:
                th, ph, ok = bloch_angles(chi[1], chi[2], tol)
            else:
                th, ph, ok = np.nan, np.nan, False
            row.update(theta=th, phi=ph, phi_defined=ok)
        rows.append(row)
        sector[ns] += weight

    out = ProjectedEnsemble(L=L, N=N, subsystem=subsystem, environment_sites=env_sites,
                            ensemble=rows, sector_weights=sector, discarded_weight=discarded)
    if two_site:
        one = [r for r in rows if r["n_system"] == 1]
        out.theta = np.array([r["theta"] for r in one])
        out.phi = np.array([r["phi"] for r in one])
        out.weights = np.array([r["weight"] for r in one])
        out.environment = [r["env"] for r in one]
    return out


# Backwards-compatible name used in the original four-site notebook.
def extract_gqs(state, L: int = 4, subsystem: Sequence[int] = (0, 1), N: int = 2,
                tol: float = 1e-12) -> ProjectedEnsemble:
    """Alias of :func:`projected_ensemble` with the four-site notebook defaults."""
    return projected_ensemble(state, L, subsystem, N=N, tol=tol)


def check_ensemble(ens: ProjectedEnsemble, psi, atol: float = 1e-10) -> None:
    """Assert the consistency checks from the original notebook.

    Total weight, normalization of each ``chi``, angle reconstruction (two-site
    case) and ``sum lambda |chi><chi| == rho_S``. Raises ``AssertionError``.
    """
    total = sum(r["weight"] for r in ens.ensemble) + ens.discarded_weight
    assert np.isclose(total, 1.0, atol=atol), "weights do not sum to one"
    for r in ens.ensemble:
        assert np.isclose(np.vdot(r["chi"], r["chi"]).real, 1.0, atol=atol)
        if len(ens.subsystem) == 2 and r["n_system"] == 1:
            rec = two_site_state_from_angles(r["theta"], r["phi"])
            assert np.allclose(np.outer(rec, rec.conj()), np.outer(r["chi"], r["chi"].conj()), atol=1e-8)
    rho = reduced_density_matrix(psi, ens.L, ens.subsystem)
    assert np.allclose(ens.density_matrix(), rho, atol=max(atol, 10 * ens.discarded_weight))
