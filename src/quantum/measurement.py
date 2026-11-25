"""Qiskit measurement circuits for count-based GQS reconstruction.

Reconstruction itself is in :mod:`chain_gqs.tomography` (NumPy only), so
hardware counts can be analysed without this module.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..tomography import SETTINGS


def gqs_measurement_circuits(evolution_circuit: QuantumCircuit,
                             subsystem: Sequence[int] = (0, 1)) -> dict[str, QuantumCircuit]:
    """``ZZ``, ``XX``, ``YX`` circuits on ``subsystem``; all other qubits in Z.

    Qubit ``i`` is measured into clbit ``i``; count strings are ``c_(L-1)...c_0``.
    """
    L = evolution_circuit.num_qubits
    if evolution_circuit.num_clbits:
        raise ValueError("provide a circuit with no classical bits")
    if any(i.operation.name in ("measure", "reset") for i in evolution_circuit.data):
        raise ValueError("input must contain preparation/evolution only")
    circuits = {}
    for setting in SETTINGS:
        qc = QuantumCircuit(L, L, name=f"gqs_{setting}")
        qc.compose(evolution_circuit, qubits=range(L), inplace=True)
        qc.barrier()
        for q, basis in zip(subsystem, setting):
            if basis == "Y":
                qc.sdg(q)
            if basis in ("X", "Y"):
                qc.h(q)
        qc.measure(range(L), range(L))
        circuits[setting] = qc
    return circuits


def sample_counts(measurement_circuits: dict, shots: int = 20000, seed=None) -> dict:
    """Ideal finite-shot sampling with ``Statevector`` (no Aer). Replace with
    device execution on hardware, preserving the qubit -> clbit mapping."""
    if not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError("shots must be a positive integer per setting")
    rng = np.random.default_rng(seed)
    counts = {}
    for setting in SETTINGS:
        qc = measurement_circuits[setting]
        state = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        state.seed(int(rng.integers(0, 2**32)))
        counts[setting] = dict(state.sample_counts(shots))
    return counts
