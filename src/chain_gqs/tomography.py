"""Reconstructing the two-site GQS from measurement counts (NumPy only).

Three settings are measured, each with the environment in Z:

======  ============================================  ========================
Setting  rotation before Z measurement                conditional expectation
======  ============================================  ========================
ZZ       none                                          z = <Z_{s0}>
XX       H on s0 and s1                                x = <X_{s0} X_{s1}>
YX       Sdg then H on s0; H on s1                     y = <Y_{s0} X_{s1}>
======  ============================================  ========================

Counts dictionaries use Qiskit's string order ``c_{L-1} ... c_0`` with qubit
``i`` measured into clbit ``i``. :func:`ideal_counts` produces such
dictionaries from a statevector without Qiskit; on hardware use the circuits
from :func:`chain_gqs.quantum.measurement.gqs_measurement_circuits`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .basis import complement, local_index, popcount
from .states import as_statevector_array

SETTINGS = ("ZZ", "XX", "YX")

_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_SDG = np.diag([1, -1j])
_ROTATION = {"Z": np.eye(2), "X": _H, "Y": _H @ _SDG}


def apply_single_qubit(psi: np.ndarray, U: np.ndarray, qubit: int, L: int) -> np.ndarray:
    """Apply a 2x2 unitary to ``qubit`` of a little-endian statevector."""
    t = psi.reshape([2] * L)            # axis 0 is qubit L-1
    ax = L - 1 - qubit
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def rotate_for_setting(psi, L: int, setting: str, subsystem: Sequence[int] = (0, 1)) -> np.ndarray:
    psi = as_statevector_array(psi)
    for q, basis in zip(subsystem, setting):
        psi = apply_single_qubit(psi, _ROTATION[basis], q, L)
    return psi


def ideal_counts(psi, L: int, subsystem: Sequence[int] = (0, 1), shots: int = 20000,
                 seed=None) -> dict[str, dict[str, int]]:
    """Ideal finite-shot counts for all three settings (no Qiskit needed)."""
    if not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError("shots must be a positive integer per setting")
    rng = np.random.default_rng(seed)
    counts = {}
    for setting in SETTINGS:
        p = np.abs(rotate_for_setting(psi, L, setting, subsystem)) ** 2
        n = rng.multinomial(shots, p / p.sum())
        counts[setting] = {format(x, f"0{L}b"): int(c) for x, c in enumerate(n) if c}
    return counts


def reconstruct_from_counts(counts_by_setting, L: int, subsystem: Sequence[int] = (0, 1),
                            N: int = 2, min_conditional_shots: int = 20,
                            assume_pure: bool = False) -> dict:
    """Reconstruct the two-site projected ensemble from counts only.

    Generalization of ``reconstruct_gqs_from_counts`` to any ``L``, ``N`` and
    pair of subsystem sites. See the original notebook for the caveats: this
    assumes fixed total ``N`` and is a simple radial-clipping estimator, not
    maximum-likelihood tomography. ``assume_pure=True`` normalizes Bloch
    vectors and builds ``chi`` for plotting.
    """
    if len(subsystem) != 2:
        raise ValueError("count reconstruction is implemented for two-site subsystems")
    if not isinstance(min_conditional_shots, (int, np.integer)) or min_conditional_shots < 1:
        raise ValueError("min_conditional_shots must be a positive integer")
    s0, s1 = subsystem
    env = complement(subsystem, L)
    ke = len(env)

    tables = {}
    for setting in SETTINGS:
        table = np.zeros((2**ke, 4), dtype=np.int64)
        for key, count in counts_by_setting[setting].items():
            bits = str(key).replace(" ", "")
            if len(bits) != L or set(bits) - {"0", "1"}:
                raise ValueError(f"count keys must be {L}-bit strings c_(L-1)...c_0")
            if not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError("counts must be nonnegative integers")
            x = int(bits, 2)            # Qiskit string -> little-endian integer
            table[local_index(x, env), 2 * ((x >> s0) & 1) + ((x >> s1) & 1)] += count
        if table.sum() == 0:
            raise ValueError(f"no shots in setting {setting}")
        tables[setting] = table

    zz = tables["ZZ"]
    total = int(zz.sum())
    leakage = sum(int(zz[a, s]) for a in range(2**ke) for s in range(4)
                  if popcount(a) + popcount(s) != N)
    if leakage:
        raise ValueError(f"{leakage}/{total} ZZ shots violate N={N}; use leakage-aware tomography")
    env_counts = {k: t.sum(axis=1) for k, t in tables.items()}
    sectors = {n: float(sum(zz[:, s].sum() for s in range(4) if popcount(s) == n) / total)
               for n in (0, 1, 2)}

    parity = np.array([1, -1, -1, 1])
    z_sign = np.array([1, 1, -1, -1])
    ensemble = []
    for a in range(2**ke):
        nz = int(env_counts["ZZ"][a])
        if nz == 0:
            continue
        ns = N - popcount(a)
        row = dict(env=format(a, f"0{ke}b"), weight=nz / total, n_system=ns,
                   theta=np.nan, phi=np.nan, phi_defined=False,
                   conditional_shots={k: int(v[a]) for k, v in env_counts.items()},
                   reconstructed=False)
        if ns in (0, 2):
            chi = np.zeros(4, dtype=complex)
            chi[0 if ns == 0 else 3] = 1
            row.update(chi=chi, reconstructed=True)
        elif ns != 1:
            row["reason"] = "subsystem sector outside 0..2"
        elif min(row["conditional_shots"].values()) >= min_conditional_shots:
            x = float(tables["XX"][a] @ parity / env_counts["XX"][a])
            y = float(tables["YX"][a] @ parity / env_counts["YX"][a])
            z = float(zz[a] @ z_sign / nz)
            raw = np.array([x, y, z])
            radius_raw = float(np.linalg.norm(raw))
            fitted = raw / max(1.0, radius_raw)
            if assume_pure and radius_raw > 1e-12:
                fitted = raw / radius_raw
            radius = float(np.linalg.norm(fitted))
            rho = np.array([[1 + fitted[2], fitted[0] - 1j * fitted[1]],
                            [fitted[0] + 1j * fitted[1], 1 - fitted[2]]]) / 2
            row.update(bloch_raw=raw, radius_raw=radius_raw, bloch=fitted,
                       radius=radius, rho_encoded=rho)
            if radius > 1e-12:
                theta = float(np.arctan2(np.hypot(x, y), z))
                defined = bool(np.hypot(x, y) > 1e-12)
                phi = float(np.arctan2(y, x) % (2 * np.pi)) if defined else 0.0
                row.update(theta=theta, phi=phi, phi_defined=defined, reconstructed=True)
                if assume_pure:
                    row["chi"] = np.array([0, np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2), 0])
            else:
                row["reason"] = "zero Bloch vector: direction undefined"
        else:
            row["reason"] = "too few conditional shots in at least one setting"
        ensemble.append(row)

    one = [r for r in ensemble if r["n_system"] == 1]
    return dict(ensemble=ensemble, sector_weights=sectors,
                theta=np.array([r["theta"] for r in one]),
                phi=np.array([r["phi"] for r in one]),
                weights=np.array([r["weight"] for r in one]),
                environment=[r["env"] for r in one],
                environment_counts=env_counts, assume_pure=bool(assume_pure),
                shots={k: int(t.sum()) for k, t in tables.items()})


def require_pure_reconstruction(result: dict) -> dict:
    """Validate a reconstruction before passing it to :func:`plot_gqs_history`."""
    if not result["assume_pure"]:
        raise ValueError("reconstruct with assume_pure=True only if the pure-state model is appropriate")
    missing = [r["env"] for r in result["ensemble"] if not r["reconstructed"]]
    if missing:
        raise ValueError(f"unresolved outcomes {missing}; increase shots")
    return result


def sampled_pauli_energy(psi, terms, L: int, shots_per_term: int = 4096, seed=11) -> dict:
    """Finite-shot estimate of ``<H>`` measuring each Pauli term separately.

    ``terms`` is a list of ``(label, coeff)`` with Qiskit label order, e.g.
    ``model.pauli_terms()``. Reports an estimated standard error.
    """
    rng = np.random.default_rng(seed)
    psi = as_statevector_array(psi)
    estimate, variance, measured = 0.0, 0.0, 0
    for label, coeff in terms:
        coeff = float(np.real(coeff))
        if set(label) == {"I"}:
            estimate += coeff
            continue
        phi, active = psi, []
        for q, p in enumerate(label[::-1]):
            if p != "I":
                active.append(q)
                phi = apply_single_qubit(phi, _ROTATION[p], q, L)
        prob = np.abs(phi) ** 2
        samples = rng.choice(2**L, size=shots_per_term, p=prob / prob.sum())
        parity = np.zeros(shots_per_term, dtype=int)
        for q in active:
            parity ^= (samples >> q) & 1
        values = 1 - 2 * parity
        estimate += coeff * values.mean()
        variance += coeff**2 * values.var(ddof=1) / shots_per_term
        measured += 1
    return {"sampled energy": estimate, "estimated standard error": float(np.sqrt(variance)),
            "total shots": measured * shots_per_term}
