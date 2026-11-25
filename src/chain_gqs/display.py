"""Reader-facing tables for a :class:`~chain_gqs.gqs.ProjectedEnsemble`.

Two styles show the same numbers in different language:

``'pe'`` (projected ensemble)
    environment configuration, n_S, n_E, lambda, p(a | n_S), |chi>, theta/pi, phi/pi.
``'cm'`` (condensed matter)
    rest-of-chain configuration, charge on the dimer, probability, and the
    conditional dimer observables: charge imbalance, bond order, bond current.

Set the default once with :func:`set_display_style`, or pass ``style=`` per call.
Tables are for reading (rounded, '—' for undefined entries); use
``ProjectedEnsemble.table()`` or the attributes for analysis.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

STYLES = ("pe", "cm")
_style = "pe"
DASH = "—"
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def set_display_style(style: str) -> None:
    """Set the default table style: ``'pe'`` or ``'cm'``."""
    global _style
    _style = _check(style)


def get_display_style() -> str:
    return _style


def _check(style: Optional[str]) -> str:
    style = _style if style is None else style
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    return style


def _sub(i: int) -> str:
    return str(i).translate(_SUB)


def config_header(sites) -> str:
    """``'|n₂n₃n₄⟩'`` for sites (2, 3, 4)."""
    return "|" + "".join(f"n{_sub(s)}" for s in sites) + "⟩"


def format_ket(chi, decimals: int = 3, tol: float = 1e-3) -> str:
    """Readable ket in the local subsystem basis; phases relative to the first term."""
    chi = np.asarray(chi)
    k = int(np.log2(len(chi)))
    idx = [i for i, c in enumerate(chi) if abs(c) > tol]
    if not idx:
        return "0"
    ref = np.angle(chi[idx[0]])
    if len(idx) == 1:
        return f"|{idx[0]:0{k}b}⟩"
    terms = []
    for i in idx:
        mag, ph = abs(chi[i]), (np.angle(chi[i]) - ref) % (2 * np.pi) / np.pi
        phase = "" if np.isclose(ph, 0, atol=10**-decimals) or np.isclose(ph, 2, atol=10**-decimals) \
            else f"·e^({ph:.2f}πi)"
        terms.append(f"{mag:.{decimals}f}{phase}|{i:0{k}b}⟩")
    return " + ".join(terms)


def _possible_sectors(ens):
    k, ke = len(ens.subsystem), len(ens.environment_sites)
    return list(range(min(k, ens.N), max(0, ens.N - ke) - 1, -1))


def _sector_description(n: int, k: int, style: str) -> str:
    if k == 2:
        pe = {0: "|00⟩", 1: "α|01⟩ + β|10⟩  (Bloch sphere)", 2: "|11⟩"}
        cm = {0: "empty dimer |00⟩", 1: "one electron shared across the bond", 2: "doubly occupied |11⟩"}
        return (pe if style == "pe" else cm)[n]
    return f"{n}-electron states of the {k}-site cluster"


def sector_table(ens, style: Optional[str] = None, decimals: int = 3):
    """Table 1: probability of each electron number on the subsystem."""
    import pandas as pd
    style = _check(style)
    k = len(ens.subsystem)
    rows = []
    for n in _possible_sectors(ens):
        count = sum(1 for r in ens.ensemble if r["n_system"] == n)
        rows.append([n, ens.N - n, round(ens.sector_weights.get(n, 0.0), decimals),
                     _sector_description(n, k, style), count])
    if style == "pe":
        cols = ["n_S", "n_E", "P(n_S)", "subsystem states", "environment configs"]
    else:
        cols = ["charge on dimer", "charge on rest of chain", "probability", "dimer state",
                "configurations"]
    df = pd.DataFrame(rows, columns=cols).set_index(cols[0])
    return df


def ensemble_table(ens, style: Optional[str] = None, include_fixed: Optional[bool] = None,
                   decimals: int = 3):
    """Table 2: one row per environment configuration.

    ``include_fixed`` controls whether rows with an empty or doubly occupied
    subsystem (whose state is fixed) are shown. Default: shown in ``'pe'``,
    hidden in ``'cm'`` (see :func:`sector_table` for their total probability).
    """
    import pandas as pd
    style = _check(style)
    include_fixed = (style == "pe") if include_fixed is None else include_fixed
    s = ens.subsystem
    two_site = len(s) == 2
    env_col = ("environment " if style == "pe" else "rest of chain ") + config_header(ens.environment_sites)

    if style == "cm":
        obs = ens.dimer_observables()   # raises unless the subsystem is a neighbouring pair
        by_env = {e: i for i, e in enumerate(obs["env"])}

    rows = sorted(ens.ensemble, key=lambda r: (-r["n_system"], r["env"]))
    out = []
    for r in rows:
        ns, one = r["n_system"], r["n_system"] == 1 and two_site
        if not include_fixed and not one:
            continue
        P_sector = ens.sector_weights[ns]
        if style == "pe":
            row = {env_col: f"|{r['env']}⟩", "n_E": ens.N - ns, "n_S": ns,
                   "λ": round(r["weight"], decimals),
                   "p(a | n_S)": round(r["weight"] / P_sector, decimals) if P_sector > 0 else DASH,
                   "|χ⟩": format_ket(r["chi"], decimals)}
            if two_site:
                row["θ/π"] = round(r["theta"] / np.pi, decimals) if one else DASH
                row["φ/π"] = round(r["phi"] / np.pi, decimals) if one and r["phi_defined"] else DASH
                row[f"P(electron on site {s[0]})"] = (
                    round(float(np.sin(r["theta"] / 2) ** 2), decimals) if one else DASH)
        else:
            row = {env_col: f"|{r['env']}⟩"}
            if include_fixed:
                row["charge on dimer"] = ns
                row["dimer state"] = {0: "empty", 1: "one electron", 2: "doubly occupied"}[ns]
            row["probability"] = round(r["weight"], decimals)
            imb = f"imbalance ⟨n{_sub(s[0])}−n{_sub(s[1])}⟩"
            bo = f"bond order ⟨c{_sub(s[0])}†c{_sub(s[1])}+h.c.⟩"
            cur = f"bond current j{_sub(s[0])}→{_sub(s[1])}/J"
            if one:
                i = by_env[r["env"]]
                row[imb] = round(float(obs["imbalance"][i]), decimals)
                row[bo] = round(float(obs["bond_order"][i]), decimals)
                row[cur] = round(float(obs["bond_current"][i]), decimals)
            else:
                # Empty or doubly occupied dimer: all three are exactly zero.
                row[imb] = row[bo] = row[cur] = 0.0
        out.append(row)
    df = pd.DataFrame(out)
    return df.set_index(env_col) if len(df) else df
