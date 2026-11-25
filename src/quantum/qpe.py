"""Quantum phase estimation with dense controlled ``exp(2 pi i (H - a) 2^j / W)``."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

from ..model import ChainModel
from ..spectrum import Spectrum, eigenspace_fidelity, energy_bounds
from .vqe import preparation


def _random_sector_angles(d: int, seed) -> list[float]:
    rng = np.random.default_rng(seed)
    amp = rng.normal(size=d)
    amp /= np.linalg.norm(amp)
    angles = [np.arctan2(np.linalg.norm(amp[j + 1:]), amp[j]) for j in range(d - 2)]
    angles.append(np.arctan2(amp[-1], amp[-2]))
    return angles


def run_qpe(model: ChainModel, m: int = 8, shots: int = 8192, seed: int = 31,
            initial_site_string: str | None = None) -> dict:
    """Run ideal QPE. Seed state: random real sector state, or a site-order string."""
    L, M = model.L, 2**m
    if initial_site_string is None:
        prep = preparation(model, _random_sector_angles(len(model.sector), seed))
    else:
        s = initial_site_string
        if len(s) != L or set(s) - {"0", "1"} or s.count("1") != model.N:
            raise ValueError(f"need an L-character site-order string with N={model.N} ones")
        prep = QuantumCircuit(L)
        for i, bit in enumerate(s):
            if bit == "1":
                prep.x(i)
    a, W = energy_bounds(model)
    qc = QuantumCircuit(m + L, m)
    system = list(range(m, m + L))
    qc.compose(prep, qubits=system, inplace=True)
    qc.h(range(m))
    for j in range(m):
        Uj = expm(2j * np.pi * (model.hamiltonian - a * np.eye(2**L)) * (2**j) / W)
        CU = np.eye(2 ** (L + 1), dtype=complex)
        CU[1::2, 1::2] = Uj          # control is least significant in [control, system...]
        qc.append(UnitaryGate(CU, label=f"ctrl-U^{2**j}"), [j] + system)
    qc.append(QFTGate(m).inverse(), range(m))
    joint = Statevector.from_instruction(qc).data.reshape(2**L, M)
    probabilities = np.sum(np.abs(joint) ** 2, axis=0)
    probabilities /= probabilities.sum()
    counts = np.random.default_rng(seed + 1).multinomial(shots, probabilities)
    measured = qc.copy()
    measured.measure(range(m), range(m))
    return dict(circuit=qc, measured_circuit=measured, joint=joint, probabilities=probabilities,
                counts=counts, grid=a + W * np.arange(M) / M, a=a, W=W, m=m,
                seed_state=Statevector.from_instruction(prep).data)


def conditional_state(qpe: dict, y: int) -> np.ndarray:
    column = qpe["joint"][:, int(y)]
    norm = np.linalg.norm(column)
    if norm < 1e-14:
        raise ValueError("outcome has negligible probability")
    return column / norm


def qpe_benchmark(model: ChainModel, qpe: dict, spectrum: Spectrum):
    """Benchmark-only table: exact levels label their nearest QPE bins."""
    import pandas as pd
    idx = model.sector
    E, vec = spectrum.energies, spectrum.vectors
    overlaps = np.abs(vec.T.conj() @ qpe["seed_state"][idx]) ** 2
    bins = np.rint((E - qpe["a"]) / qpe["W"] * 2 ** qpe["m"]).astype(int)
    rows = []
    for k, y in enumerate(bins):
        fid = eigenspace_fidelity(conditional_state(qpe, y)[idx], spectrum, k)
        rows.append([k, E[k], qpe["grid"][y], overlaps[k], qpe["probabilities"][y], fid])
    return pd.DataFrame(rows, columns=["level", "exact", "QPE", "overlap", "probability", "fidelity"])


def qpe_top_outcomes(qpe: dict, n: int = 12):
    import pandas as pd
    top = np.argsort(qpe["counts"])[-n:][::-1]
    return pd.DataFrame({"phase bits": [format(y, f"0{qpe['m']}b") for y in top],
                         "energy estimate": qpe["grid"][top], "counts": qpe["counts"][top]})


def qpe_kernel_probabilities(qpe: dict, spectrum: Spectrum, sector: np.ndarray) -> np.ndarray:
    """Analytic finite-register distribution ``p(y)``, used only as a test."""
    M = 2 ** qpe["m"]
    phi = (spectrum.energies - qpe["a"]) / qpe["W"]
    w = np.abs(spectrum.vectors.T.conj() @ qpe["seed_state"][sector]) ** 2
    delta = phi[:, None] - np.arange(M)[None, :] / M
    kernel = np.abs(np.mean(np.exp(2j * np.pi * delta[:, :, None] * np.arange(M)), axis=2)) ** 2
    return w @ kernel
