from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class RegularizedLogistic:
    """Static-only constrained member of the rank-one probe model family."""

    def __init__(self, *, l2: float = 1.0, max_iter: int = 500) -> None:
        if l2 <= 0 or max_iter < 1:
            raise ValueError("l2 and max_iter must be positive")
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.parameters_: np.ndarray | None = None
        self.converged_: bool = False

    def fit(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RegularizedLogistic":
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or len(np.unique(y)) != 2:
            raise ValueError("fit expects a finite matrix and two-class labels")
        if not np.isfinite(x).all():
            raise ValueError("static values must be finite")
        weight = (
            np.ones(len(x), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        if weight.shape != (len(x),) or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("sample weights must be nonnegative and nonzero")
        weight = weight / weight.sum()

        def objective(parameters: np.ndarray):
            intercept, coefficient = parameters[0], parameters[1:]
            score = intercept + x @ coefficient
            residual = (expit(score) - y) * weight
            loss = np.sum(weight * (np.logaddexp(0.0, score) - y * score))
            loss += 0.5 * self.l2 * float(coefficient @ coefficient)
            gradient = np.concatenate(
                [[residual.sum()], x.T @ residual + self.l2 * coefficient]
            )
            return float(loss), gradient

        prevalence = float(np.average(y, weights=weight))
        start = np.zeros(x.shape[1] + 1, dtype=np.float64)
        start[0] = np.log((prevalence + 1e-4) / (1.0 - prevalence + 1e-4))
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-10},
        )
        if (
            not result.success
            or not np.isfinite(result.fun)
            or not np.isfinite(result.x).all()
        ):
            raise RuntimeError(f"static optimization did not converge: {result.message}")
        self.parameters_ = np.asarray(result.x, dtype=np.float64)
        self.converged_ = bool(result.success)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(values, dtype=np.float64)
        return expit(self.parameters_[0] + x @ self.parameters_[1:])

    @property
    def coefficients(self) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("model is not fitted")
        return self.parameters_.copy()


class RankOneLogistic:
    """Logistic probe with one separable tensor coefficient and a linear control head."""

    def __init__(
        self,
        *,
        l2: float = 1.0,
        restarts: int = 3,
        max_iter: int = 500,
        seed: int = 17,
    ) -> None:
        if l2 <= 0 or restarts < 1 or max_iter < 1:
            raise ValueError("l2, restarts, and max_iter must be positive")
        self.l2 = float(l2)
        self.restarts = int(restarts)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.shape_: tuple[int, int, int] | None = None
        self.parameters_: np.ndarray | None = None
        self.static_dim_: int = 0
        self.converged_: bool = False
        self.objective_: float | None = None
        self.baseline_objective_: float | None = None
        self.used_baseline_: bool = False
        self.signal_restarts_converged_: int = 0

    def _unpack(self, parameters: np.ndarray):
        if self.shape_ is None:
            raise RuntimeError("model shape is unavailable")
        a, b, c = self.shape_
        start = 1
        gamma = parameters[start : start + self.static_dim_]
        start += self.static_dim_
        u = parameters[start : start + a]
        start += a
        v = parameters[start : start + b]
        start += b
        w = parameters[start : start + c]
        return parameters[0], gamma, u, v, w

    def _initial_factors(self, tensor: np.ndarray, labels: np.ndarray):
        correlation = np.mean(
            tensor * (labels - labels.mean())[:, None, None, None], axis=0
        )
        if not np.any(np.abs(correlation) > 1e-14):
            return tuple(np.zeros(size, dtype=np.float64) for size in correlation.shape)
        u = np.linalg.svd(
            correlation.reshape(correlation.shape[0], -1), full_matrices=False
        )[0][:, 0]
        remainder = np.einsum("abc,a->bc", correlation, u)
        left, _, right = np.linalg.svd(remainder, full_matrices=False)
        return u, left[:, 0], right[0]

    def fit(
        self,
        tensor: np.ndarray,
        labels: np.ndarray,
        static: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        baseline_parameters: np.ndarray | None = None,
    ) -> "RankOneLogistic":
        x = np.asarray(tensor, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 4 or y.shape != (len(x),) or len(np.unique(y)) != 2:
            raise ValueError("fit expects [sample,a,b,c] and two-class labels")
        controls = (
            np.zeros((len(x), 0))
            if static is None
            else np.asarray(static, dtype=np.float64)
        )
        if (
            controls.shape[0] != len(x)
            or controls.ndim != 2
            or not np.isfinite(controls).all()
        ):
            raise ValueError("static controls must be a finite [sample,feature] matrix")
        weight = (
            np.ones(len(x))
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        if weight.shape != (len(x),) or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("sample weights must be nonnegative and nonzero")
        weight = weight / weight.sum()
        self.shape_ = tuple(int(value) for value in x.shape[1:])
        self.static_dim_ = int(controls.shape[1])
        baseline = None
        if baseline_parameters is not None:
            baseline = np.asarray(baseline_parameters, dtype=np.float64)
            if baseline.shape != (self.static_dim_ + 1,) or not np.isfinite(
                baseline
            ).all():
                raise ValueError(
                    "baseline parameters must contain one intercept and one "
                    "coefficient per static feature"
                )
        base_factors = self._initial_factors(x, y)
        rng = np.random.default_rng(self.seed)

        def objective(parameters: np.ndarray):
            intercept, gamma, u, v, w = self._unpack(parameters)
            score = intercept + controls @ gamma + np.einsum(
                "nabc,a,b,c->n", x, u, v, w, optimize=True
            )
            residual = (expit(score) - y) * weight
            penalty = np.concatenate([gamma, u, v, w])
            loss = np.sum(weight * (np.logaddexp(0.0, score) - y * score))
            loss += 0.5 * self.l2 * float(penalty @ penalty)
            gradient = [float(residual.sum())]
            gradient.extend((controls.T @ residual + self.l2 * gamma).tolist())
            gradient.extend(
                (np.einsum("n,nabc,b,c->a", residual, x, v, w) + self.l2 * u).tolist()
            )
            gradient.extend(
                (np.einsum("n,nabc,a,c->b", residual, x, u, w) + self.l2 * v).tolist()
            )
            gradient.extend(
                (np.einsum("n,nabc,a,b->c", residual, x, u, v) + self.l2 * w).tolist()
            )
            return float(loss), np.asarray(gradient)

        prevalence = float(np.average(y, weights=weight))
        static_start = (
            baseline
            if baseline is not None
            else np.concatenate(
                [
                    [np.log((prevalence + 1e-4) / (1.0 - prevalence + 1e-4))],
                    np.zeros(self.static_dim_),
                ]
            )
        )
        starts = []
        for restart in range(self.restarts):
            factors = base_factors if restart == 0 else tuple(
                rng.normal(scale=0.2, size=size) for size in self.shape_
            )
            starts.append(np.concatenate([static_start, *factors]))
        results = [
            minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": 1e-10},
            )
            for start in starts
        ]
        converged = [
            result
            for result in results
            if result.success
            and np.isfinite(result.fun)
            and np.isfinite(result.x).all()
        ]
        self.signal_restarts_converged_ = len(converged)
        baseline_vector = None
        baseline_objective = None
        if baseline is not None:
            baseline_vector = np.concatenate(
                [baseline, *(np.zeros(size, dtype=np.float64) for size in self.shape_)]
            )
            baseline_objective = float(objective(baseline_vector)[0])
        if not converged and baseline_vector is None:
            messages = "; ".join(str(result.message) for result in results)
            raise RuntimeError(f"no rank-one restart converged: {messages}")
        best = (
            min(converged, key=lambda result: float(result.fun))
            if converged
            else None
        )
        use_baseline = baseline_vector is not None and (
            best is None or float(best.fun) >= baseline_objective - 1e-12
        )
        if use_baseline:
            self.parameters_ = baseline_vector
            self.objective_ = baseline_objective
        else:
            if best is None:
                raise RuntimeError("rank-one optimization produced no valid candidate")
            self.parameters_ = np.asarray(best.x, dtype=np.float64)
            self.objective_ = float(best.fun)
        self.baseline_objective_ = baseline_objective
        self.used_baseline_ = bool(use_baseline)
        self.converged_ = True
        return self

    def decision_function(
        self, tensor: np.ndarray, static: np.ndarray | None = None
    ) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(tensor, dtype=np.float64)
        controls = (
            np.zeros((len(x), 0))
            if static is None
            else np.asarray(static, dtype=np.float64)
        )
        intercept, gamma, u, v, w = self._unpack(self.parameters_)
        return intercept + controls @ gamma + np.einsum(
            "nabc,a,b,c->n", x, u, v, w, optimize=True
        )

    def predict_proba(self, tensor: np.ndarray, static: np.ndarray | None = None) -> np.ndarray:
        return expit(self.decision_function(tensor, static))

    @property
    def coefficient_tensor(self) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("model is not fitted")
        _, _, u, v, w = self._unpack(self.parameters_)
        return np.einsum("a,b,c->abc", u, v, w)

    @property
    def model_parameters(self) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("model is not fitted")
        return self.parameters_.copy()

    @property
    def signal_parameters(self) -> int:
        if self.shape_ is None:
            raise RuntimeError("model is not fitted")
        return int(sum(self.shape_) - 2)
