"""Plotting. Every function takes data and an optional ``ax`` and returns axes;
nothing calls ``plt.show()``, so figures can be saved or combined freely."""
from __future__ import annotations

import numpy as np

DEFAULT_CMAP_COLORS = ["#ffb3ba", "#ed5068", "#a50026"]


def _default_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pink_red", DEFAULT_CMAP_COLORS)


def _sites(item, sites=None):
    """Subsystem sites for labels: explicit, else from the ensemble, else (0, 1)."""
    if sites is not None:
        return tuple(sites)
    sub = getattr(item, "subsystem", None)
    if sub is None and isinstance(item, dict):
        sub = item.get("subsystem")
    return tuple(sub) if sub is not None and len(sub) == 2 else (0, 1)


def _angle_axes(ax, sites=(0, 1)):
    """theta-phi axes labelled in both languages.

    Left axis: theta. Right axis: charge imbalance <n_a - n_b> = -cos(theta).
    Bottom: phi, with 0 = bonding and pi = antibonding (equator, hopping -J < 0).
    """
    a, b = sites
    ax.set(xlim=(0, 2 * np.pi), ylim=(0, np.pi),
           xlabel=r"$\phi$  (relative phase across the bond)",
           ylabel=r"$\theta$")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0\nbonding", r"$\pi/2$", "$\\pi$\nantibonding", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    sec = ax.secondary_yaxis(
        "right", functions=(lambda t: -np.cos(t), lambda y: np.arccos(-np.clip(y, -1, 1))))
    sec.set_ticks([-1, 0, 1], [f"−1\n(on site {b})", "0", f"+1\n(on site {a})"])
    sec.set_ylabel(rf"charge imbalance $\langle n_{a} - n_{b}\rangle$")
    return sec


def plot_gqs(gqs, ax=None, cmap="Reds", s=100, sites=None):
    """Scatter ``phi`` vs ``theta`` of one ensemble; colour = unconditional weight.
    Axes carry both the angle and the charge-imbalance / bonding labels."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    pts = ax.scatter(gqs["phi"], gqs["theta"], c=gqs["weights"], s=s, cmap=cmap,
                     vmin=0, vmax=1, edgecolors="black")
    _angle_axes(ax, _sites(gqs, sites))
    ax.figure.colorbar(pts, ax=ax, pad=0.15, label=r"probability $\lambda_a$")
    return ax


def _snapshot(item, sector_history_row=None):
    """Return ``(points[M,3] = (theta, phi, w), probs[3])`` for one snapshot."""
    rows = item["ensemble"] if (hasattr(item, "ensemble") or isinstance(item, dict)) else item
    tagged = all("n_system" in r for r in rows)
    if not tagged and sector_history_row is None:
        raise ValueError("Include n_system for every state, or supply sector_history")
    probs, points = np.zeros(3), []
    for r in rows:
        w = float(r["weight"] if "weight" in r else r["lambda"])
        ns = r["n_system"] if tagged else 1
        if ns not in (0, 1, 2) or not np.isfinite(w) or w < 0:
            raise ValueError("invalid sector or probability weight")
        probs[ns] += w
        if ns == 1 and w > 0:
            th, ph = float(r["theta"]), float(r["phi"])
            if not np.all(np.isfinite([th, ph])) or not 0 <= th <= np.pi:
                raise ValueError("Bloch angles must be finite with theta in [0, pi]")
            points.append((th, ph % (2 * np.pi), w))
    if sector_history_row is not None:
        supplied = np.asarray(sector_history_row, float)
        if tagged and not np.allclose(supplied, probs, atol=1e-8, rtol=0):
            raise ValueError("sector_history disagrees with the full ensemble")
        if not tagged and not np.isclose(supplied[1], probs[1], atol=1e-8, rtol=0):
            raise ValueError("Bloch weights must sum to P_one (use unconditional weights)")
        probs = supplied.copy()
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("sector probabilities must be nonnegative and sum to one")
    return np.asarray(points).reshape(-1, 3), probs / probs.sum()


def draw_bloch_sphere(ax, sites=(0, 1)):
    """Bloch sphere of the one-electron dimer, labelled in both languages.

    x = bond order <c_a^dag c_b + h.c.>, y = -j_(a->b)/J (bond current),
    z = cos(theta) = <n_b - n_a> (north pole: electron on site b).
    """
    a, b = sites
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    ax.plot_wireframe(np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v),
                      color="0.75", alpha=0.25, linewidth=0.5)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), zlim=(-1.15, 1.15))
    ax.set_xlabel(f"x: bond order", labelpad=2)
    ax.set_ylabel(rf"y: $-j_{{{a}\to{b}}}/J$ (current)", labelpad=2)
    ax.set_zlabel(rf"z = cos$\theta$ = $\langle n_{b}-n_{a}\rangle$", labelpad=2)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.text(0, 0, 1.15, rf"$|01\rangle$: electron on site {b}", ha="center", fontsize=8)
    ax.text(0, 0, -1.3, rf"$|10\rangle$: electron on site {a}", ha="center", fontsize=8)
    ax.text(1.25, 0, 0, "bonding", ha="center", fontsize=8)
    ax.text(-1.3, 0, 0, "antibonding", ha="center", fontsize=8)
    ax.grid(False)


def draw_sector_simplex(ax):
    vertices = np.array([[0, 0], [0.5, np.sqrt(3) / 2], [1, 0]])
    outline = vertices[[0, 1, 2, 0]]
    ax.plot(outline[:, 0], outline[:, 1], color="0.4", lw=1)
    ax.text(0, -0.045, "$|00\\rangle$\nempty dimer\n($n_S=0$)", ha="center", va="top", fontsize=9)
    ax.text(1, -0.045, "$|11\\rangle$\ndoubly occupied\n($n_S=2$)", ha="center", va="top", fontsize=9)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "one electron on dimer ($n_S=1$)\n(Bloch sphere)",
            ha="center", va="bottom", fontsize=9)
    ax.set(xlim=(-0.15, 1.15), ylim=(-0.25, 1.12), title="Charge on the dimer (sector probabilities)")
    ax.set_aspect("equal")
    ax.axis("off")
    return vertices


def plot_gqs_history(Q_S, kick=None, past_marker_size=30, current_marker_size=120,
                     past_alpha=0.2, vmin=0.0, vmax=1.0, show_colorbar=True,
                     bloch_view="sphere", sector_history=None, cmap=None,
                     figsize=(12, 5), title=None, sites=None):
    """Conditional Bloch states (left) and sector-probability simplex (right).

    ``Q_S`` is a sequence of :class:`~chain_gqs.gqs.ProjectedEnsemble` objects,
    dicts with an ``'ensemble'`` key (e.g. from
    :func:`~chain_gqs.tomography.reconstruct_from_counts`) or lists of rows.
    ``kick`` is a 1-based snapshot index (default: last). Returns
    ``fig, (ax_bloch, ax_simplex)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    kick = len(Q_S) if kick is None else kick
    if not isinstance(kick, (int, np.integer)) or not 1 <= kick <= len(Q_S):
        raise ValueError(f"kick must be an integer between 1 and {len(Q_S)}")
    if bloch_view not in ("sphere", "angles"):
        raise ValueError("bloch_view must be 'sphere' or 'angles'")
    if not (0 <= past_alpha <= 1 and vmin < vmax):
        raise ValueError("require 0 <= past_alpha <= 1 and vmin < vmax")
    if sector_history is not None:
        sector_history = np.asarray(sector_history, float)
        if sector_history.shape != (len(Q_S), 3):
            raise ValueError("sector_history must have shape (len(Q_S), 3)")
    cmap = _default_cmap() if cmap is None else cmap
    norm = Normalize(vmin, vmax)
    snaps = [_snapshot(Q_S[n], None if sector_history is None else sector_history[n])
             for n in range(kick)]

    fig = plt.figure(figsize=figsize, layout="constrained")
    ax_b = fig.add_subplot(121, projection="3d" if bloch_view == "sphere" else None)
    ax_s = fig.add_subplot(122)
    lab_sites = _sites(Q_S[0], sites)
    if bloch_view == "sphere":
        draw_bloch_sphere(ax_b, lab_sites)
    else:
        _angle_axes(ax_b, lab_sites)
    ax_b.set_title("Conditional dimer states (one electron on dimer)")
    vertices = draw_sector_simplex(ax_s)

    for n, (points, probs) in enumerate(snaps):
        current = n == kick - 1
        size = current_marker_size if current else past_marker_size
        alpha = 1.0 if current else past_alpha
        edge = "black" if current else "none"
        if len(points):
            th, ph, w = points.T
            if bloch_view == "sphere":
                args = (np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))
                extra = {"depthshade": False}
            else:
                args, extra = (ph, th), {}
            ax_b.scatter(*args, c=w, cmap=cmap, norm=norm, s=size, alpha=alpha,
                         edgecolors=edge, linewidths=0.5, **extra)
        ax_s.scatter(*(probs @ vertices), color="#4682B4", s=size, alpha=alpha,
                     edgecolors=edge, linewidths=0.5, zorder=3 if current else 2)
    if show_colorbar:
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_b, shrink=0.7,
                     pad=0.07 if bloch_view == "angles" else 0.12,
                     label=r"probability $\lambda_a$")
    if title:
        fig.suptitle(title)
    return fig, (ax_b, ax_s)


def plot_sector_weights(times, Q_S, ax=None):
    """``P(N_S = n)`` versus time for each subsystem sector."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    P = np.array([[q["sector_weights"][n] for n in sorted(q["sector_weights"])] for q in Q_S])
    for n in range(P.shape[1]):
        ax.plot(times, P[:, n], label=fr"$P(n_S={n})$: {['empty', 'one electron', 'doubly occupied'][n] if P.shape[1] == 3 else ''}")
    ax.set(xlabel=r"time $t$ (units of $1/J$)", ylabel="probability of charge on dimer", ylim=(-0.02, 1.02))
    ax.legend()
    return ax


def plot_qpe_histogram(qpe: dict, exact_energies=None, ax=None, title=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    width = qpe["W"] / 2 ** qpe["m"]
    ax.bar(qpe["grid"], qpe["counts"] / qpe["counts"].sum(), width=0.85 * width)
    if exact_energies is not None:
        for k, e in enumerate(exact_energies):
            ax.axvline(e, color="crimson", alpha=0.5, label="Exact energies" if k == 0 else None)
        ax.legend()
    ax.set(xlabel="Energy estimate", ylabel="Sample frequency", title=title)
    return ax


def plot_vqe_convergence(trace, exact_ground=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.minimum.accumulate(trace), label="Best VQE energy so far")
    if exact_ground is not None:
        ax.axhline(exact_ground, color="crimson", linestyle="--", label="Exact ground energy")
    ax.set(xlabel="Objective evaluation (best restart)", ylabel="Energy")
    ax.legend()
    return ax
