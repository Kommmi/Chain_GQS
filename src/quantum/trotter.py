"""Qiskit Trotter circuits for the chain (any L)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..basis import index_from_occupied
from ..gqs import ProjectedEnsemble, projected_ensemble
from ..model import ChainModel


def trotter_step(model: ChainModel, dt: float) -> QuantumCircuit:
    """One first-order step: hopping bonds left to right, then ``cp`` for every
    pair interaction, then ``p`` for site energies.

    ``rxx(-J dt)`` and ``ryy(-J dt)`` on a bond together implement
    ``exp(+i J dt (XX + YY) / 2)`` and conserve particle number.
    Equals :func:`chain_gqs.dynamics.trotter_propagator` gate for gate.
    """
    L, J = model.L, model.J
    Vij, eps = model.interactions, model.site_energies
    qc = QuantumCircuit(L, name="dt")
    for i in range(L - 1):
        qc.rxx(-J * dt, i, i + 1)
        qc.ryy(-J * dt, i, i + 1)
    for i in range(L):
        for j in range(i + 1, L):
            if Vij[i, j] != 0:
                qc.cp(-Vij[i, j] * dt, i, j)
    for i in range(L):
        if eps[i] != 0:
            qc.p(-eps[i] * dt, i)
    return qc


def preparation_circuit(L: int, occupied: Sequence[int]) -> QuantumCircuit:
    index_from_occupied(occupied, L)  # validates
    qc = QuantumCircuit(L)
    for i in occupied:
        qc.x(i)
    return qc


def build_circuit(model: ChainModel, nsteps: int, dt: float,
                  occupied: Sequence[int] = (0, 2)) -> QuantumCircuit:
    """Prepare ``occupied`` and evolve to ``t = nsteps * dt``; no measurements."""
    if not isinstance(nsteps, (int, np.integer)) or nsteps < 0:
        raise ValueError("nsteps must be a nonnegative integer")
    if len(occupied) != model.N:
        raise ValueError(f"occupied must list N={model.N} sites")
    qc = preparation_circuit(model.L, occupied)
    step = trotter_step(model, dt)
    for _ in range(nsteps):
        qc.compose(step, inplace=True)
    return qc


def trotter_gqs_history(model: ChainModel, nsteps: int, dt: float,
                        occupied: Sequence[int] = (0, 2), subsystem: Sequence[int] = (0, 1),
                        tol: float = 1e-12) -> tuple[np.ndarray, list[ProjectedEnsemble]]:
    """Circuit-based equivalent of :func:`chain_gqs.dynamics.gqs_history` (method='trotter')."""
    state = Statevector.from_instruction(preparation_circuit(model.L, occupied))
    step = trotter_step(model, dt)
    Q = [projected_ensemble(state, model.L, subsystem, N=model.N, tol=tol)]
    for _ in range(nsteps):
        state = state.evolve(step)
        Q.append(projected_ensemble(state, model.L, subsystem, N=model.N, tol=tol))
    return np.arange(nsteps + 1) * dt, Q
