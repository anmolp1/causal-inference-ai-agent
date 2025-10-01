from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

# Optional heavy deps: load lazily
try:
	from econml.dr import DRLearner  # type: ignore
	econml_available = True
except Exception:  # pragma: no cover - optional
	econml_available = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


@dataclass
class CausalInferenceConfig:
	random_state: int = 42
	n_estimators: int = 200
	max_depth: Optional[int] = None
	min_samples_leaf: int = 5
	cate_clip: Optional[Tuple[float, float]] = None
	bootstrap_samples: int = 500
	bootstrap_ci: float = 0.95


@dataclass
class CausalEstimate:
	ate: float
	ate_ci: Optional[Tuple[float, float]]
	cate: np.ndarray
	uplift: np.ndarray
	meta: Dict[str, Any]


class CausalInferenceAgent:
	"""CATE and uplift modeling wrapper.
	Uses DR-Learner if econml is available; otherwise falls back to T-learner with
	RandomForestRegressor.
	"""

	def __init__(self, config: Optional[CausalInferenceConfig] = None) -> None:
		self.config = config or CausalInferenceConfig()
		self._fitted = False
		self._model: Any = None
		self._t_model: Any = None
		self._c_model: Any = None

	def _fit_dr(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> None:
		final_model = RandomForestRegressor(
			n_estimators=self.config.n_estimators,
			max_depth=self.config.max_depth,
			min_samples_leaf=self.config.min_samples_leaf,
			random_state=self.config.random_state,
		)
		model_propensity = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
		model_regression = RandomForestRegressor(
			n_estimators=self.config.n_estimators,
			max_depth=self.config.max_depth,
			min_samples_leaf=self.config.min_samples_leaf,
			random_state=self.config.random_state,
		)
		self._model = DRLearner(
			model_propensity=model_propensity,
			model_regression=model_regression,
			model_final=final_model,
			random_state=self.config.random_state,
		)
		self._model.fit(y, w, X=X)

	def _fit_tlearner(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> None:
		# Two separate outcome models
		self._t_model = RandomForestRegressor(
			n_estimators=self.config.n_estimators,
			max_depth=self.config.max_depth,
			min_samples_leaf=self.config.min_samples_leaf,
			random_state=self.config.random_state,
		)
		self._c_model = RandomForestRegressor(
			n_estimators=self.config.n_estimators,
			max_depth=self.config.max_depth,
			min_samples_leaf=self.config.min_samples_leaf,
			random_state=self.config.random_state,
		)
		self._t_model.fit(X[w == 1], y[w == 1])
		self._c_model.fit(X[w == 0], y[w == 0])

	def fit(self, X: ArrayLike, w: ArrayLike, y: ArrayLike) -> None:
		X = np.asarray(X)
		w = np.asarray(w).astype(int)
		y = np.asarray(y).astype(float)

		if econml_available:
			self._fit_dr(X, w, y)
		else:
			self._fit_tlearner(X, w, y)

		self._fitted = True

	def predict_cate(self, X: ArrayLike) -> np.ndarray:
		assert self._fitted, "Model not fitted. Call fit(...) first."
		X = np.asarray(X)
		if econml_available:
			cate = self._model.effect(X)
		else:
			y1 = self._t_model.predict(X)
			y0 = self._c_model.predict(X)
			cate = y1 - y0

		if self.config.cate_clip is not None:
			low, high = self.config.cate_clip
			cate = np.clip(cate, low, high)
		return cate

	def estimate(self, X: ArrayLike, w: ArrayLike, y: ArrayLike) -> CausalEstimate:
		self.fit(X, w, y)
		cate = self.predict_cate(X)
		uplift = cate.copy()
		ate = float(np.mean(cate))
		ci = self._bootstrap_ate_ci(X, w, y)
		return CausalEstimate(ate=ate, ate_ci=ci, cate=cate, uplift=uplift, meta={})

	def _bootstrap_ate_ci(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
		alpha = 1.0 - self.config.bootstrap_ci
		n = X.shape[0]
		if n < 50:
			return None
		ates: List[float] = []
		rng = np.random.default_rng(self.config.random_state)
		for _ in range(self.config.bootstrap_samples):
			idx = rng.integers(0, n, size=n)
			Xi, wi, yi = X[idx], w[idx], y[idx]
			try:
				if econml_available:
					self._fit_dr(Xi, wi, yi)
					cate_i = self._model.effect(Xi)
				else:
					self._fit_tlearner(Xi, wi, yi)
					y1 = self._t_model.predict(Xi)
					y0 = self._c_model.predict(Xi)
					cate_i = y1 - y0
				ates.append(float(np.mean(cate_i)))
			except Exception:
				continue
		if not ates:
			return None
		lo = float(np.quantile(ates, alpha / 2))
		hi = float(np.quantile(ates, 1 - alpha / 2))
		return (lo, hi)

	def uplift_auc(self, X: ArrayLike, w: ArrayLike, y: ArrayLike) -> Optional[float]:
		"""Proxy metric using treatment interaction ranking for binary y.
		Computes AUC on transformed labels for a rough check.
		"""
		try:
			X = np.asarray(X)
			w = np.asarray(w).astype(int)
			y = np.asarray(y).astype(int)
			s = self.predict_cate(X)
			# Knightly transformation: label 1 for treated responders, 0 otherwise
			z = ((w == 1) & (y == 1)).astype(int)
			return float(roc_auc_score(z, s))
		except Exception:
			return None
