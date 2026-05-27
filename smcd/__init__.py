"""SMCD: Step-level Manifold Constraint Diagnostic."""

from .features import load_geometry, FeatureConfig
from .constraint_score import ConstraintScore
from .transition_kernel import TransitionKernel
from .probe import ConstraintProbe
from .detector import CUSUMDetector, conformal_pvalues
from .dataset import SMCDDataset, collate_sequences
