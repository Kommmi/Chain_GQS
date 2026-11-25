"""Display tables and the dimer observables behind the 'cm' style."""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from chain_gqs import (ChainModel, evolve, product_state, projected_ensemble,  # noqa: E402
                       random_sector_state, townsend2021)
from chain_gqs.basis import occupations  # noqa: E402
from chain_gqs.display import format_ket, get_display_style, set_display_style  # noqa: E402


def _hop(L, a, b):
    """Full-space matrix of c_a^dag c_b for neighbouring a, b (no JW sign)."""
    M = np.zeros((2**L, 2**L))
    for x in range(2**L):
        if (x >> b) & 1 and not (x >> a) & 1:
            M[x ^ (1 << a) ^ (1 << b), x] = 1
    return M


@pytest.mark.parametrize("L,sub", [(4, (0, 1)), (5, (0, 1)), (5, (2, 3)), (5, (3, 2))])
def test_weighted_observables_equal_expectation_values(L, sub):
    psi = random_sector_state(L, 2, seed=7, real=False)
    ens = projected_ensemble(psi, L, sub)
    obs = ens.dimer_observables()
    a, b = sub
    n = np.array([occupations(x, L) for x in range(2**L)])
    p = np.abs(psi) ** 2
    hop = _hop(L, a, b)
    exp_imb = p @ (n[:, a] - n[:, b])
    exp_bo = np.vdot(psi, (hop + hop.T) @ psi).real
    exp_cur = -np.vdot(psi, 1j * (hop - hop.T) @ psi).real
    w = obs["weight"]
    assert np.isclose(w @ obs["imbalance"], exp_imb)
    assert np.isclose(w @ obs["bond_order"], exp_bo)
    assert np.isclose(w @ obs["bond_current"], exp_cur)


def test_current_sign_matches_continuity():
    # Two sites only: dn_1/dt must equal j_{0->1}.
    m = ChainModel(L=2, N=1, J=0.8)
    psi0 = np.array([0, np.cos(0.3), np.exp(0.7j) * np.sin(0.3), 0])
    dt = 1e-6
    traj = evolve(m, psi0, 1, dt)
    n1 = [np.sum(np.abs(s[[2, 3]]) ** 2) for s in traj]
    ens = projected_ensemble(psi0, 2, (0, 1))
    j = m.J * ens.dimer_observables()["bond_current"][0]
    assert np.isclose((n1[1] - n1[0]) / dt, j, rtol=1e-4)


def test_tables_both_styles_consistent():
    m = townsend2021(L=5, lam_ee=1.0)
    psi = evolve(m, product_state(5, (0, 2)), 100, 0.02)[-1]
    ens = projected_ensemble(psi, 5, (0, 1))
    pe_s, cm_s = ens.sector_table("pe"), ens.sector_table("cm")
    assert np.allclose(pe_s["P(n_S)"].to_numpy(), cm_s["probability"].to_numpy())
    assert np.isclose(pe_s["P(n_S)"].sum(), 1, atol=2e-3)

    pe = ens.ensemble_table("pe")
    assert len(pe) == len(ens.ensemble) and "NaN" not in pe.to_string()
    cm = ens.ensemble_table("cm")
    assert len(cm) == len(ens.theta)               # one-electron rows only by default
    assert len(ens.ensemble_table("cm", include_fixed=True)) == len(ens.ensemble)
    one = pe[pe["n_S"] == 1]
    assert np.allclose(one["λ"].to_numpy(), cm["probability"].to_numpy())
    # P(electron on site 0) = (1 + imbalance)/2
    imb = cm[[c for c in cm.columns if c.startswith("imbalance")][0]].to_numpy()
    assert np.allclose(one["P(electron on site 0)"].to_numpy(), (1 + imb) / 2, atol=2e-3)


def test_default_style_and_validation():
    ens = projected_ensemble(random_sector_state(4, 2, seed=1), 4)
    old = get_display_style()
    try:
        set_display_style("cm")
        assert "probability" in ens.sector_table().columns
        with pytest.raises(ValueError):
            set_display_style("xyz")
    finally:
        set_display_style(old)


def test_cm_requires_neighbouring_pair():
    ens = projected_ensemble(random_sector_state(5, 2, seed=1), 5, (0, 2))
    ens.ensemble_table("pe")
    with pytest.raises(ValueError):
        ens.ensemble_table("cm")


def test_format_ket():
    assert format_ket(np.array([0, 1, 0, 0])) == "|01⟩"
    k = format_ket(np.array([0, 0.6, 0.8j, 0]))
    assert k == "0.600|01⟩ + 0.800·e^(0.50πi)|10⟩"
