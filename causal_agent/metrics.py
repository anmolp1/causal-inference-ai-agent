from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def uplift_at_k(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k_frac: float = 0.3) -> float:
	"""Compute net uplift at top-k fraction using transformed outcome method.
	Args:
		uplift_scores: predicted per-customer uplift (higher is better)
		treatment: binary treatment indicators
		outcome: binary outcomes
		k_frac: fraction of population to target
	Returns:
		Estimated net uplift@k (treated responders minus control responders)
	"""
	n = len(uplift_scores)
	if n == 0:
		return 0.0
	k = max(1, int(np.floor(k_frac * n)))
	order = np.argsort(-uplift_scores)
	idx = order[:k]
	t = treatment[idx].astype(int)
	y = outcome[idx].astype(int)
	# Net uplift in top-k: responders among treated minus responders among control (scaled by prevalence)
	resp_t = int(np.sum((t == 1) & (y == 1)))
	resp_c = int(np.sum((t == 0) & (y == 1)))
	return float(resp_t - resp_c)


def qini_curve(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute Qini curve coordinates.
	Returns cumulative population fraction and incremental gains arrays.
	"""
	order = np.argsort(-uplift_scores)
	t = treatment[order].astype(int)
	y = outcome[order].astype(int)
	n = len(uplift_scores)
	if n == 0:
		return np.zeros(1), np.zeros(1)
	frac = np.arange(1, n + 1) / n
	# Cumulative responders among treated and control up to each k
	cum_t = np.cumsum((t == 1) & (y == 1))
	cum_c = np.cumsum((t == 0) & (y == 1))
	# Qini = diff between treated and control responders
	qini = cum_t - cum_c
	return frac, qini.astype(float)


def qini_auc(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
	frac, q = qini_curve(uplift_scores, treatment, outcome)
	if len(frac) < 2:
		return 0.0
	# Area under Qini curve via trapezoidal rule
	return float(np.trapz(q, frac))


def auuc(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
	"""Area under uplift curve using transformed outcome z = 2y-1 for treated and - for control.
	This is a simple proxy variant.
	"""
	order = np.argsort(-uplift_scores)
	t = treatment[order].astype(int)
	y = outcome[order].astype(int)
	z = np.where(t == 1, y, -y)
	cum = np.cumsum(z)
	n = len(z)
	if n == 0:
		return 0.0
	frac = np.arange(1, n + 1) / n
	return float(np.trapz(cum, frac))


def evaluate_uplift(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k_fracs: List[float] = [0.1, 0.2, 0.3]) -> Dict[str, float]:
	metrics: Dict[str, float] = {}
	metrics["qini_auc"] = qini_auc(uplift_scores, treatment, outcome)
	metrics["auuc"] = auuc(uplift_scores, treatment, outcome)
	for k in k_fracs:
		metrics[f"uplift_at_{int(k*100)}"] = uplift_at_k(uplift_scores, treatment, outcome, k_frac=k)
	return metrics


