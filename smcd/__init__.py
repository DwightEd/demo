"""SMCD v2: Grassmannian Spectral Trajectory Diagnostic.

Core idea: model the step-by-step evolution of reasoning subspaces on the
Grassmannian, combined with spectral shape tracking. Anomaly = deviation
from the learned conditional density of correct trajectories.

Modules:
    grassmann       — Gr(k,d) operations: Log map, distance, tangent PCA
    representation  — Build transition vectors t_j = [v_j, sigma_j, delta_sigma_j]
    density         — Conditional density model p(t_j | t_{<j})
    detector        — CUSUM + conformal calibration for sequence-level detection
"""

from .grassmann import grassmann_log, grassmann_distance, principal_angles, TangentPCA
from .representation import load_subspaces, learn_tangent_pca, compute_representations
from .density import ConditionalDensity
from .detector import CUSUMDetector, conformal_pvalues
