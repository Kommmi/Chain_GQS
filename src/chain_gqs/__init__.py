"""chain_gqs: projected ensembles (geometric quantum states) of spinless
electrons on a 1D Coulomb chain.

Core modules are NumPy-only. Qiskit circuits live in :mod:`chain_gqs.quantum`.
"""
from .basis import sector_indices, site_label
from .dynamics import evolve, exact_propagator, gqs_history, trotter_propagator
from .gqs import (ProjectedEnsemble, bloch_angles, bloch_vector, check_ensemble, extract_gqs,
                  number_distribution, projected_ensemble, reduced_density_matrix)
from .model import ChainModel, coulomb_matrix, nearest_neighbor_matrix
from .presets import PRESETS, core_potential, coulomb, screened_coulomb_matrix, townsend2021
from .spectrum import Spectrum, energy_bounds, exact_spectrum
from .states import product_state, random_sector_state, state_from_label
from .tomography import ideal_counts, reconstruct_from_counts, require_pure_reconstruction

from .display import get_display_style, set_display_style

__version__ = "0.1.0"
