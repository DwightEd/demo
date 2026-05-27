"""CUSUM sequential detector + conformal calibration.

CUSUM: S_j = max(0, S_{j-1} + s_j - k)
  - k = median(s_cal) + 0.5 * std(s_cal) from calibration set
  - Natural decay when surprise drops → self-correction robust

Conformal p-value: p_j = (1 + |{s_cal >= s_j}|) / (n_cal + 1)
"""

import numpy as np
from typing import List, Tuple, Optional


def conformal_pvalues(
    scores: np.ndarray,
    cal_scores: np.ndarray,
) -> np.ndarray:
    """Convert scores to conformal p-values using calibration set.

    p_j = (1 + #{s_cal >= s_j}) / (n_cal + 1)
    """
    n_cal = len(cal_scores)
    pvals = np.array([
        (1 + np.sum(cal_scores >= s)) / (n_cal + 1) for s in scores
    ])
    return pvals


class CUSUMDetector:
    """CUSUM change-point detector for step-level scores.

    Calibrate on correct trajectories, then detect on new sequences.
    """

    def __init__(self, k: float = None, threshold: float = 5.0):
        """
        Args:
            k: reference value (auto-calibrated if None)
            threshold: CUSUM alarm threshold
        """
        self.k = k
        self.threshold = threshold
        self.cal_scores = None

    def calibrate(self, correct_scores: List[np.ndarray]):
        """Calibrate from correct trajectory scores.

        Args:
            correct_scores: list of [T_i] arrays of per-step scores from correct trajectories
        """
        all_scores = np.concatenate(correct_scores)
        self.cal_scores = all_scores

        if self.k is None:
            self.k = float(np.median(all_scores) + 0.5 * np.std(all_scores))

    def cusum(self, scores: np.ndarray) -> np.ndarray:
        """Run CUSUM on a single sequence.

        Returns:
            cusum_values: [T] cumulative sum statistics
        """
        T = len(scores)
        S = np.zeros(T)
        for j in range(T):
            prev = S[j - 1] if j > 0 else 0.0
            S[j] = max(0.0, prev + scores[j] - self.k)
        return S

    def detect(
        self, scores: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """Full detection: CUSUM + conformal p-values.

        Args:
            scores: [T] per-step scores for one example
            alpha: significance level for conformal test

        Returns:
            cusum_values: [T] CUSUM statistics
            pvalues: [T] conformal p-values
            alarm_step: first step where CUSUM > threshold, or None
        """
        cusum_values = self.cusum(scores)

        pvalues = conformal_pvalues(scores, self.cal_scores) if self.cal_scores is not None else np.ones(len(scores))

        alarm_indices = np.where(cusum_values > self.threshold)[0]
        alarm_step = int(alarm_indices[0]) if len(alarm_indices) > 0 else None

        return cusum_values, pvalues, alarm_step


def evaluate_detection(
    detector: CUSUMDetector,
    score_sequences: List[np.ndarray],
    labels: List[int],
    alpha: float = 0.05,
) -> dict:
    """Evaluate CUSUM detector on a set of examples.

    Args:
        score_sequences: list of [T_i] score arrays
        labels: list of int (-1 = correct, else first error step)

    Returns:
        dict with TP, FP, TN, FN, precision, recall, F1, detection_delay
    """
    TP = FP = TN = FN = 0
    delays = []

    for scores, label in zip(score_sequences, labels):
        _, _, alarm = detector.detect(scores, alpha)

        if label == -1:
            # Correct trajectory
            if alarm is None:
                TN += 1
            else:
                FP += 1
        else:
            # Has error at step `label`
            if alarm is not None:
                TP += 1
                delays.append(max(0, alarm - label))
            else:
                FN += 1

    n = TP + FP + TN + FN
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_delay = float(np.mean(delays)) if delays else None

    return {
        "n": n, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
        "avg_detection_delay": avg_delay,
    }
