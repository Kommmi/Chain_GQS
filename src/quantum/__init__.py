"""Qiskit-dependent parts of :mod:`chain_gqs`. Importing this requires Qiskit;
the rest of the package does not."""
from .measurement import gqs_measurement_circuits, sample_counts
from .qpe import conditional_state, qpe_benchmark, qpe_kernel_probabilities, qpe_top_outcomes, run_qpe
from .trotter import build_circuit, preparation_circuit, trotter_gqs_history, trotter_step
from .vqe import preparation, shot_energy, variational_benchmark, variational_spectrum

__all__ = [
    "trotter_step", "build_circuit", "preparation_circuit", "trotter_gqs_history",
    "gqs_measurement_circuits", "sample_counts",
    "run_qpe", "conditional_state", "qpe_benchmark", "qpe_top_outcomes", "qpe_kernel_probabilities",
    "preparation", "variational_spectrum", "variational_benchmark", "shot_energy",
]
