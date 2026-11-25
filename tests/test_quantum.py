"""Qiskit-dependent checks; skipped automatically if Qiskit is not installed."""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector, partial_trace  # noqa: E402

from chain_gqs import ChainModel, evolve, exact_spectrum, product_state, projected_ensemble  # noqa: E402
from chain_gqs.quantum import (build_circuit, gqs_measurement_circuits, qpe_kernel_probabilities,  # noqa: E402
                               run_qpe, sample_counts, trotter_step, variational_spectrum)
from chain_gqs.tomography import ideal_counts, reconstruct_from_counts  # noqa: E402


@pytest.mark.parametrize("L", [4, 5])
def test_circuit_equals_numpy_trotter(L):
    m = ChainModel(L=L, J=0.7, V=1.3, eps=np.linspace(0, 0.5, L))
    psi_c = Statevector.from_instruction(build_circuit(m, 40, 0.02, occupied=(0, 2))).data
    psi_n = evolve(m, product_state(L, (0, 2)), 40, 0.02, "trotter")[-1]
    assert np.allclose(psi_c, psi_n, atol=1e-10)


def test_pauli_op_matches_dense():
    m = ChainModel(L=5, V=2.0)
    assert np.allclose(m.sparse_pauli_op().to_matrix(), m.hamiltonian)


def test_reduced_state_matches_qiskit_partial_trace():
    m = ChainModel(L=4)
    psi = Statevector.from_instruction(build_circuit(m, 100, 0.02))
    ens = projected_ensemble(psi, 4, (0, 1))
    order = [0, 2, 1, 3]   # Qiskit |q1 q0> -> site order |q0 q1>
    assert np.allclose(ens.density_matrix(), partial_trace(psi, [2, 3]).data[np.ix_(order, order)])


def test_measurement_circuits_match_numpy_sampler():
    m = ChainModel(L=5)
    circ = build_circuit(m, 30, 0.02)
    psi = Statevector.from_instruction(circ).data
    a = reconstruct_from_counts(sample_counts(gqs_measurement_circuits(circ), 100_000, seed=1), 5, assume_pure=True)
    b = reconstruct_from_counts(ideal_counts(psi, 5, shots=100_000, seed=2), 5, assume_pure=True)
    for ra, rb in zip(a["ensemble"], b["ensemble"]):
        assert ra["env"] == rb["env"] and abs(ra["weight"] - rb["weight"]) < 1e-2


@pytest.mark.parametrize("L", [4, 5])
def test_qpe_and_vqe(L):
    m = ChainModel(L=L, V=2.0)
    spec = exact_spectrum(m)
    qpe = run_qpe(m, m=6, seed=35)
    assert np.allclose(qpe_kernel_probabilities(qpe, spec, m.sector), qpe["probabilities"], atol=1e-10)
    vqe = variational_spectrum(m, restarts=2, seed=76, verbose=False)
    assert np.max(abs(vqe["table"]["energy"].to_numpy() - spec.energies)) < 1e-4
