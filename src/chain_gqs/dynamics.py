"""Time evolution (NumPy only).

Two propagators are provided:

* :func:`exact_propagator` -- ``exp(-i H dt)`` from diagonalization.
* :func:`trotter_propagator` -- the same first-order product formula as the
  Qiskit circuit in :mod:`chain_gqs.quantum.trotter` (hopping bonds left to
  right, then pair interactions, then site potentials). Use it to reproduce
  circuit results without Qiskit, or to measure Trotter error.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .basis import occupations
from .gqs import ProjectedEnsemble, projected_ensemble
from .model import ChainModel
from .states import as_statevector_array


def exact_propagator(model: ChainModel, dt: float) -> np.ndarray:
    E, U = np.linalg.eigh(model.hamiltonian)
    return (U * np.exp(-1j * E * dt)) @ U.conj().T


def _bond_hopping(L: int, i: int, J: float) -> np.ndarray:
    H = np.zeros((2**L, 2**L))
    for x in range(2**L):
        if ((x >> i) & 1) != ((x >> (i + 1)) & 1):
            H[x ^ (1 << i) ^ (1 << (i + 1)), x] = -J
    return H


def trotter_propagator(model: ChainModel, dt: float) -> np.ndarray:
    """One first-order Trotter step, gate-for-gate equal to the Qiskit circuit."""
    L = model.L
    U = np.eye(2**L, dtype=complex)
    for i in range(L - 1):
        E, W = np.linalg.eigh(_bond_hopping(L, i, model.J))
        U = ((W * np.exp(-1j * E * dt)) @ W.T) @ U
    Vij, eps = model.interactions, model.site_energies
    iu = np.triu_indices(L, 1)
    diag = np.empty(2**L)
    for x in range(2**L):
        n = occupations(x, L)
        diag[x] = np.sum(Vij[iu] * n[iu[0]] * n[iu[1]]) + eps @ n
    return np.exp(-1j * diag * dt)[:, None] * U


PROPAGATORS: dict[str, Callable[[ChainModel, float], np.ndarray]] = {
    "exact": exact_propagator,
    "trotter": trotter_propagator,
}


def evolve(model: ChainModel, psi0, nsteps: int, dt: float, method: str = "exact") -> np.ndarray:
    """Return the trajectory, shape ``(nsteps + 1, 2**L)``, at times ``k * dt``."""
    if method not in PROPAGATORS:
        raise ValueError(f"method must be one of {list(PROPAGATORS)}")
    if not isinstance(nsteps, (int, np.integer)) or nsteps < 0:
        raise ValueError("nsteps must be a nonnegative integer")
    U = PROPAGATORS[method](model, dt)
    psi = as_statevector_array(psi0)
    if psi.size != model.dim:
        raise ValueError("initial state has the wrong dimension")
    out = np.empty((nsteps + 1, model.dim), dtype=complex)
    out[0] = psi
    for k in range(nsteps):
        psi = U @ psi
        out[k + 1] = psi
    return out


def gqs_history(model: ChainModel, psi0, nsteps: int, dt: float,
                subsystem: Sequence[int] = (0, 1), method: str = "exact",
                tol: float = 1e-12) -> tuple[np.ndarray, list[ProjectedEnsemble]]:
    """Evolve and extract the projected ensemble at every step.

    Replaces the copy-pasted loop of the original notebook:

    >>> times, Q_S = gqs_history(ChainModel(L=4, J=0.5), product_state(4, (0, 2)), 300, 0.02)
    """
    traj = evolve(model, psi0, nsteps, dt, method)
    times = np.arange(nsteps + 1) * dt
    return times, [projected_ensemble(psi, model.L, subsystem, N=model.N, tol=tol) for psi in traj]
