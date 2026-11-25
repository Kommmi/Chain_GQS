"""VQE / VQD with a number-preserving chain of two-level rotations."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from ..model import ChainModel
from ..spectrum import Spectrum, eigenspace_fidelity, energy_bounds
from ..tomography import sampled_pauli_energy


def preparation(model: ChainModel, theta) -> QuantumCircuit:
    """Rotate successive sector configurations; covers all real sector states.

    Needs ``d - 1`` angles, ``d`` = sector dimension.
    """
    L, idx = model.L, model.sector
    if len(theta) != len(idx) - 1:
        raise ValueError("need d-1 angles")
    qc = QuantumCircuit(L)
    for i in range(L):
        if (idx[0] >> i) & 1:
            qc.x(i)
    for j, angle in enumerate(theta):
        a, b = idx[j:j + 2]
        R = np.eye(2**L)
        c, s = np.cos(angle), np.sin(angle)
        R[a, a] = R[b, b] = c
        R[b, a], R[a, b] = s, -s
        qc.append(UnitaryGate(R, label=f"R{j}({angle:.2f})"), range(L))
    return qc


def variational_spectrum(model: ChainModel, n_states: int | None = None, restarts: int = 3,
                         seed: int = 72, verbose: bool = True) -> dict:
    """VQE for the ground state and VQD for ``n_states - 1`` excited states."""
    import pandas as pd
    d = len(model.sector)
    n_states = d if n_states is None else n_states
    if not 1 <= n_states <= d:
        raise ValueError("n_states must lie between 1 and the sector dimension")
    H = model.hamiltonian
    beta = 2 * energy_bounds(model)[1]
    rng = np.random.default_rng(seed)
    states, records, circuits, traces = [], [], [], []
    for k in range(n_states):
        best = None
        for _ in range(restarts):
            trace = []

            def objective(theta):
                psi = Statevector.from_instruction(preparation(model, theta)).data
                cost = np.vdot(psi, H @ psi).real + beta * sum(abs(np.vdot(o, psi)) ** 2 for o in states)
                trace.append(cost)
                return cost

            res = minimize(objective, rng.uniform(-np.pi, np.pi, d - 1), method="BFGS",
                           options={"gtol": 1e-7, "maxiter": 400})
            if best is None or res.fun < best[0].fun:
                best = (res, trace)
        res, trace = best
        circuit = preparation(model, res.x)
        psi = Statevector.from_instruction(circuit).data
        energy = np.vdot(psi, H @ psi).real
        records.append(dict(level=k, energy=energy,
                            residual=np.linalg.norm(H @ psi - energy * psi),
                            leakage=1 - np.sum(np.abs(psi[model.sector]) ** 2),
                            optimizer_success=bool(res.success), message=str(res.message)))
        states.append(psi); circuits.append(circuit); traces.append(trace)
        if verbose:
            print(f"{'VQE' if k == 0 else 'VQD'} level {k}: E={energy:.8f}, residual={records[-1]['residual']:.2e}")
    return dict(table=pd.DataFrame(records), states=np.array(states), circuits=circuits, traces=traces)


def variational_benchmark(model: ChainModel, result: dict, spectrum: Spectrum):
    table = result["table"].copy()
    n = len(table)
    E = spectrum.energies
    table["exact E"] = E[:n]
    table["abs energy error"] = abs(table["energy"] - E[:n])
    table["eigenspace fidelity"] = [eigenspace_fidelity(s[model.sector], spectrum, k)
                                    for k, s in enumerate(result["states"])]
    return table


def shot_energy(model: ChainModel, prep: QuantumCircuit, shots_per_term: int = 4096, seed: int = 11) -> dict:
    """Finite-shot energy of a prepared circuit (per-term Pauli measurement)."""
    psi = Statevector.from_instruction(prep).data
    return sampled_pauli_energy(psi, model.pauli_terms(), model.L, shots_per_term, seed)
