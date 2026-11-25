# chain_gqs

Projected ensembles (geometric quantum states, GQS) of spinless electrons on an
open 1D chain with Coulomb interaction. This package replaces the code in
`four_site_gqs.ipynb` and `Electron_Chain_Classical_QPE_VQE.ipynb` and works
for **any** chain length `L`, electron number `N` and choice of subsystem sites.

## Install (editable, so changes to the code apply right away)

```bash
cd chain_gqs
pip install -e ".[dev]"     # or: pip install -e .   (NumPy/SciPy/Matplotlib only)
pytest                       # Qiskit tests are skipped if Qiskit is missing
```

## Layout

```
src/chain_gqs/
  basis.py       bit-order conventions, N-electron sectors, subsystem/environment split
  model.py       ChainModel (L, N, J, V, eps, Vij) -> dense H, sector H, Pauli terms
  presets.py     named parameter sets: coulomb (original notebooks), townsend2021 (paper)
  states.py      product_state, state_from_label, random_sector_state
  dynamics.py    exact and Trotter propagators, evolve(), gqs_history()
  gqs.py         projected_ensemble(), ProjectedEnsemble, Bloch angles, checks
  spectrum.py    exact_spectrum(), Gershgorin energy_bounds(), fidelities
  tomography.py  count-based reconstruction (+ a NumPy shot sampler)
  plotting.py    plot_gqs, plot_gqs_history, plot_sector_weights, QPE/VQE plots
  quantum/       everything that needs Qiskit
    trotter.py       trotter_step, build_circuit, trotter_gqs_history
    measurement.py   ZZ/XX/YX measurement circuits, sample_counts
    qpe.py           run_qpe, qpe_benchmark, ...
    vqe.py           preparation, variational_spectrum (VQE/VQD), shot_energy
tests/           consistency checks from the notebooks, now automated
examples/        scripts and notebooks that only *use* the package
```

Rules followed throughout:

* **Physics in modules, experiments in notebooks.** Notebooks set parameters,
  call functions, and plot. They define no functions of their own.
* **Computation and display are separate.** Functions return arrays, dataclasses
  or DataFrames; plotting functions take an `ax` and never call `plt.show()`.
* **One model object.** `ChainModel` holds all parameters and is passed
  everywhere, rather than repeating `J, V, epsilon, interactions` in every
  signature. `model.with_(V=2.0)` makes a modified copy for sweeps.
* **Qiskit is optional.** GQS extraction, dynamics and count reconstruction are
  NumPy only; `dynamics.trotter_propagator` is gate-for-gate equal to the
  Qiskit circuit (tested in `tests/test_quantum.py`).

## Conventions

Site `i` is qubit `i`; statevector index `x = sum_i n_i 2^i` (Qiskit order).
Labels are written in increasing site order `|n_0 n_1 ... >`. For a list of
sites (subsystem or environment) the **first listed site is the most
significant bit** of the local index, so subsystem `(0, 1)` uses
`[|00>, |01>, |10>, |11>]` and `theta = 0` is `|01>`, `theta = pi` is `|10>`,
the same as the original notebook.

## Typical use

```python
from chain_gqs import ChainModel, product_state, gqs_history, projected_ensemble
from chain_gqs.plotting import plot_gqs_history

model = ChainModel(L=5, N=2, J=0.5, V=1.0)        # or townsend2021(L=5, lam_ee=1.0)
times, Q = gqs_history(model, product_state(5, (0, 2)), nsteps=300, dt=0.02,
                       subsystem=(0, 1), method="trotter")   # or "exact"
fig, axes = plot_gqs_history(Q, bloch_view="angles")

Q[-1].table()            # DataFrame of env, weight, n_system, theta, phi
Q[-1].sector_weights     # {0: P(N_S=0), 1: P(N_S=1), 2: P(N_S=2)}
```

Measurement-based reconstruction (hardware counts or simulated):

```python
from chain_gqs.quantum import build_circuit, gqs_measurement_circuits, sample_counts
from chain_gqs import reconstruct_from_counts, require_pure_reconstruction

circ = build_circuit(model, nsteps=100, dt=0.02, occupied=(0, 2))
counts = sample_counts(gqs_measurement_circuits(circ, subsystem=(0, 1)), shots=10_000, seed=42)
res = require_pure_reconstruction(reconstruct_from_counts(counts, L=5, subsystem=(0, 1), assume_pure=True))
```

Spectra:

```python
from chain_gqs import exact_spectrum
from chain_gqs.quantum import run_qpe, qpe_benchmark, variational_spectrum, variational_benchmark

spec = exact_spectrum(model.with_(V=2.0))
```

## Model presets

`ChainModel` only knows `J`, `Vij` and `eps`. A preset turns physical parameters
into those and records them on the model, so sweeps rebuild everything consistently.

```python
from chain_gqs import townsend2021, coulomb

m = townsend2021(L=5, N=2, lam_ee=1.0)   # Townsend et al., arXiv:2011.07113
m.with_(lam_ee=10.0)                      # rescales Vij AND the core potential eps
m.label                                   # 'townsend2021(L=5, N=2, lam_ee=1, t=1, ...)'

coulomb(L=5, J=1.0, V=1.0)                # V/|i-j|, the original notebooks' model
```

`townsend2021` (paper Eqs. 1-3), defaults as in the paper:

| Argument | Meaning | Default |
|---|---|---|
| `lam_ee` | e-e Coulomb strength; sweep `lam_ee / t` | 1.0 |
| `t` | hopping (becomes `J`) | 1.0 |
| `f_ex` | nearest-neighbour exchange reduction | 0.2 |
| `zeta_ee`, `zeta_nuc` | orbital cutoffs (lattice spacings) | 0.5 |
| `lam_nuc` | core attraction; `None` means equal to `lam_ee` (assumption: the paper says they scale together) | `None` |
| `Z` | core charge per site; `None` means `N/L` (charge neutral) | `None` |
| `include_self` | include own core in the potential sum (uniform shift only) | `True` |

`V_ij = lam_ee (1 - f_ex d_{|i-j|,1}) / (|i-j| + zeta_ee)` and
`eps_i = -lam_nuc Z sum_j 1/(|i-j| + zeta_nuc)`.

`model.with_(...)` on a preset accepts only that preset's parameters (so
`with_(V=...)` on a `townsend2021` model raises). On a hand-built model with an
explicit `Vij`, changing `V` also raises, because it would have no effect.

To add a model variant, write a factory in `presets.py` decorated with
`@register_preset("name")` that returns `ChainModel(..., preset="name", params=...)`.

## Mapping from the old notebooks

| Old (notebook)                                  | New                                                         |
|-------------------------------------------------|-------------------------------------------------------------|
| `trotter_step(dt, J, V, epsilon, interactions)` | `quantum.trotter_step(model, dt)`                            |
| `build_circuit(nsteps, dt, J, V, ..., occupied)`| `quantum.build_circuit(model, nsteps, dt, occupied)`         |
| `extract_gqs(state)` (L=4, sites 0,1 only)      | `projected_ensemble(state, L, subsystem, N)` (`extract_gqs` alias kept) |
| copy-pasted history loop per `J`                | `gqs_history(...)` / `quantum.trotter_gqs_history(...)`      |
| `plot_gqs`, `plot_gqs_history`                  | `plotting.plot_gqs`, `plotting.plot_gqs_history`             |
| `build_gqs_measurement_circuits`                | `quantum.gqs_measurement_circuits(circ, subsystem)`          |
| `sample_gqs_counts`                             | `quantum.sample_counts` or NumPy `tomography.ideal_counts`   |
| `reconstruct_gqs_from_counts` (L=4 only)        | `tomography.reconstruct_from_counts(counts, L, subsystem, N)`|
| `measurement_gqs_for_plot`                      | `tomography.require_pure_reconstruction`                     |
| `make_model(L, J, V, eps, Vij)` dict            | `ChainModel(...)`: `.hamiltonian`, `.sector_hamiltonian`, `.pauli_terms()`, `.sparse_pauli_op()` |
| `classical(model)` (computes + displays)        | `exact_spectrum(model)` → `Spectrum` (`.table()`, `.vectors_table()`) |
| `qpe_report`, `variational_report`              | `qpe_benchmark` / `variational_benchmark` (tables) + `plotting.plot_qpe_histogram` / `plot_vqe_convergence` |
| `shot_energy`                                   | `quantum.shot_energy` (Qiskit) or `tomography.sampled_pauli_energy` |
| assertion cells                                 | `tests/`, plus `check_ensemble(ens, psi)` for use in notebooks |

## Adding new analysis

Put measures of geometry/complexity of the ensemble (e.g. entropies of the
ensemble distribution on the Bloch sphere, trajectory statistics) in a new
module such as `src/chain_gqs/analysis.py` that takes `ProjectedEnsemble`
objects or the `Q` list from `gqs_history`, and add a test for it.
