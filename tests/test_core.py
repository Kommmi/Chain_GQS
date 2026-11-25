"""Checks ported from both notebooks (NumPy-only parts)."""
import numpy as np
import pytest

from chain_gqs import (ChainModel, check_ensemble, evolve, exact_spectrum, gqs_history,
                       ideal_counts, product_state, projected_ensemble, random_sector_state,
                       reconstruct_from_counts, reduced_density_matrix)
from chain_gqs.model import pauli_terms_matrix
from chain_gqs.spectrum import free_fermion_energies
from chain_gqs.tomography import sampled_pauli_energy


@pytest.mark.parametrize("L", [3, 4, 5, 6])
def test_hamiltonian_matches_pauli_and_conserves_number(L):
    m = ChainModel(L=L, J=1.0, V=2.0, eps=np.linspace(-0.3, 0.4, L))
    assert np.allclose(m.hamiltonian, pauli_terms_matrix(m.pauli_terms()))
    assert m.conserves_number()


@pytest.mark.parametrize("L", [4, 5])
def test_free_fermion_limit(L):
    assert np.allclose(exact_spectrum(ChainModel(L=L, V=0.0)).energies, free_fermion_energies(L, 2))


def test_four_site_matches_original_index_convention():
    # |1010> in site order; env (2,3) holds one electron on site 2 -> env label '10'.
    psi = product_state(4, (0, 2))
    ens = projected_ensemble(psi, 4, (0, 1))
    (row,) = ens.ensemble
    assert row["env"] == "10" and row["n_system"] == 1
    assert np.isclose(row["theta"], np.pi)   # electron on site 0 -> |10> -> south pole


@pytest.mark.parametrize("L,subsystem", [(4, (0, 1)), (5, (0, 1)), (5, (1, 2)), (5, (3, 1)), (6, (2, 3))])
def test_ensemble_reconstructs_reduced_state(L, subsystem):
    m = ChainModel(L=L, J=1.0, V=1.5)
    traj = evolve(m, product_state(L, (0, 2)), 50, 0.05)
    for psi in traj[::10]:
        ens = projected_ensemble(psi, L, subsystem, N=2)
        check_ensemble(ens, psi)
        assert np.isclose(sum(ens.sector_weights.values()) + ens.discarded_weight, 1)
        # Number of Bloch points <= number of env patterns with one electron.
        assert len(ens.theta) <= L - 2


def test_three_site_subsystem_has_no_bloch_angles():
    psi = random_sector_state(5, 2, seed=1)
    ens = projected_ensemble(psi, 5, (0, 1, 2))
    check_ensemble(ens, psi)
    assert "theta" not in ens.ensemble[0]


def test_rejects_leakage():
    psi = (product_state(4, (0, 2)) + product_state(4, (0,))) / np.sqrt(2)
    with pytest.raises(ValueError):
        projected_ensemble(psi, 4, N=2)


def test_trotter_converges_to_exact():
    m = ChainModel(L=5, J=1.0, V=1.0)
    psi0 = product_state(5, (0, 2))
    errs = []
    for dt, n in [(0.04, 50), (0.02, 100), (0.01, 200)]:
        a = evolve(m, psi0, n, dt, "trotter")[-1]
        b = evolve(m, psi0, n, dt, "exact")[-1]
        errs.append(np.linalg.norm(a - b))
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] / errs[2] == pytest.approx(2, rel=0.15)   # first order


def test_gqs_history_shapes():
    times, Q = gqs_history(ChainModel(L=5, J=0.5), product_state(5, (0, 2)), 20, 0.02)
    assert len(times) == len(Q) == 21 and np.isclose(times[-1], 0.4)


@pytest.mark.parametrize("L,subsystem", [(4, (0, 1)), (5, (0, 1)), (5, (2, 3))])
def test_counts_reconstruction_close_to_exact(L, subsystem):
    m = ChainModel(L=L, J=1.0, V=1.0)
    psi = evolve(m, product_state(L, (0, 2)), 60, 0.02)[-1]
    exact = projected_ensemble(psi, L, subsystem)
    rec = reconstruct_from_counts(ideal_counts(psi, L, subsystem, shots=200_000, seed=3),
                                  L, subsystem, assume_pure=True)
    exact_rows = {r["env"]: r for r in exact.ensemble if r["weight"] > 1e-3}
    for r in rec["ensemble"]:
        if r["env"] in exact_rows:
            e = exact_rows[r["env"]]
            assert r["weight"] == pytest.approx(e["weight"], abs=5e-3)
            if r["n_system"] == 1 and e["weight"] > 0.05:
                rec_v = np.array([np.sin(r["theta"]) * np.cos(r["phi"]),
                                  np.sin(r["theta"]) * np.sin(r["phi"]), np.cos(r["theta"])])
                ex_v = np.array([np.sin(e["theta"]) * np.cos(e["phi"]),
                                 np.sin(e["theta"]) * np.sin(e["phi"]), np.cos(e["theta"])])
                assert np.linalg.norm(rec_v - ex_v) < 0.05


def test_sampled_energy_unbiased():
    m = ChainModel(L=4, V=2.0)
    spec = exact_spectrum(m)
    psi = spec.full_vector(0, 4)
    out = sampled_pauli_energy(psi, m.pauli_terms(), 4, shots_per_term=20000, seed=0)
    assert abs(out["sampled energy"] - spec.energies[0]) < 5 * out["estimated standard error"] + 1e-9


def test_reduced_density_matrix_trace():
    psi = random_sector_state(5, 2, seed=4, real=False)
    assert np.isclose(np.trace(reduced_density_matrix(psi, 5, (1, 3))), 1)
