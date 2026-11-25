"""Presets: values from Townsend et al. (arXiv:2011.07113) and sweep behaviour."""
import numpy as np
import pytest

from chain_gqs import ChainModel, coulomb, evolve, product_state, projected_ensemble, townsend2021
from chain_gqs.basis import site_label
from chain_gqs.model import pauli_terms_matrix


def test_interaction_values():
    V = townsend2021(L=5, lam_ee=1.0).interactions
    # (1 - 0.2)/1.5, 1/2.5, 1/3.5, 1/4.5
    assert np.allclose(V[0, 1:], [0.8 / 1.5, 1 / 2.5, 1 / 3.5, 1 / 4.5])
    assert np.allclose(V, V.T) and np.allclose(np.diag(V), 0)


@pytest.mark.parametrize("L,expected", [
    (4, [-1.6762, -1.8667, -1.8667, -1.6762]),
    (5, [-1.4298, -1.6076, -1.6533, -1.6076, -1.4298]),
])
def test_core_potential_values(L, expected):
    assert np.allclose(townsend2021(L=L, N=2, lam_ee=1.0).site_energies, expected, atol=1e-4)


def test_lowest_configurations():
    for L, lowest in [(4, "0110"), (5, "01010")]:
        m = townsend2021(L=L, lam_ee=1.0)
        diag = np.diag(m.hamiltonian)[m.sector]
        assert site_label(m.sector[np.argmin(diag)], L) == lowest


def test_lam_nuc_defaults_to_lam_ee_and_scales_together():
    m = townsend2021(L=4, lam_ee=1.0)
    m5 = m.with_(lam_ee=5.0)
    assert np.allclose(m5.interactions, 5 * m.interactions)
    assert np.allclose(m5.site_energies, 5 * m.site_energies)
    assert m5.preset == "townsend2021" and m5.params["lam_ee"] == 5.0
    assert np.allclose(townsend2021(L=4, lam_ee=1.0, lam_nuc=1.0).hamiltonian, m.hamiltonian)


def test_independent_lam_nuc_and_f_ex():
    m = townsend2021(L=4, lam_ee=2.0, lam_nuc=0.0, f_ex=0.0)
    assert np.allclose(m.site_energies, 0)
    assert np.isclose(m.interactions[0, 1], 2.0 / 1.5)


def test_with_rejects_parameters_not_in_preset():
    with pytest.raises(ValueError, match="lam_ee"):
        townsend2021(L=4).with_(V=3.0)


def test_with_guards_explicit_Vij():
    m = ChainModel(L=4, Vij=np.ones((4, 4)) - np.eye(4))
    with pytest.raises(ValueError):
        m.with_(V=2.0)
    m.with_(J=0.5)   # other changes still fine


def test_coulomb_preset_matches_plain_model():
    a = coulomb(L=5, J=0.5, V=1.0)
    b = ChainModel(L=5, J=0.5, V=1.0)
    assert np.allclose(a.hamiltonian, b.hamiltonian)
    assert np.allclose(a.with_(V=2.0).hamiltonian, b.with_(V=2.0).hamiltonian)


def test_preset_through_full_pipeline():
    m = townsend2021(L=5, lam_ee=3.0)
    assert m.conserves_number()
    assert np.allclose(m.hamiltonian, pauli_terms_matrix(m.pauli_terms()))
    psi = evolve(m, product_state(5, (0, 2)), 50, 0.02, "trotter")[-1]
    ens = projected_ensemble(psi, 5, (0, 1))
    assert np.isclose(sum(ens.sector_weights.values()) + ens.discarded_weight, 1)


def test_label():
    assert townsend2021(L=4, lam_ee=5).label.startswith("townsend2021(L=4, N=2, lam_ee=5")
    assert ChainModel(L=4).label == "L=4, N=2, J=1, V=1"
