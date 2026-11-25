"""Sweep the interaction strength for 4 and 5 sites and save GQS history plots.

Uses the Townsend et al. (arXiv:2011.07113) parameters. Swap the preset line for
``coulomb(L=L, J=1.0, V=1.0)`` and sweep ``V`` to get the original model.

Run:  python examples/compare_interaction_strength.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chain_gqs import gqs_history, product_state, townsend2021
from chain_gqs.plotting import plot_gqs_history, plot_sector_weights

dt, nsteps = 0.02, 300
lam_values = [0.5, 2.0, 10.0]

for L in (4, 5):
    base = townsend2021(L=L, N=2, lam_ee=1.0, t=1.0)
    for lam in lam_values:
        model = base.with_(lam_ee=lam)          # rescales Vij and eps together
        times, Q = gqs_history(model, product_state(L, (0, 2)), nsteps, dt,
                               subsystem=(0, 1), method="trotter")
        tag = f"L{L}_lam{lam:g}"
        fig, _ = plot_gqs_history(Q, bloch_view="angles", title=f"L={L}, lam_ee/t={lam:g}")
        fig.savefig(f"gqs_{tag}.png", dpi=110)
        plt.close(fig)
        ax = plot_sector_weights(times, Q)
        ax.set_title(f"L={L}, lam_ee/t={lam:g}")
        ax.figure.savefig(f"sectors_{tag}.png", dpi=110, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"{model.label}: final sector weights",
              {k: round(v, 3) for k, v in Q[-1].sector_weights.items()})
