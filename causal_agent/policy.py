from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


import numpy as np




@dataclass
class ActionSpec:
    action_id: str
    cost: float = 0.0
    label: str = "offer"
    eligible: bool = True


@dataclass
class PolicyConfig:
    min_uplift: float = 0.0
    min_ate: Optional[float] = None
    confidence_ate_positive: float = 0.9
    budget_fraction: Optional[float] = None  # if set, cap top fraction by uplift
    actions: Optional[List[ActionSpec]] = None  # action catalog




class NextBestActionPolicy:
	"""Policy to select customers for intervention based on uplift and significance.
	"""


	def __init__(self, config: Optional[PolicyConfig] = None) -> None:
		self.config = config or PolicyConfig()


    def recommend(
        self,
        uplift: np.ndarray,
        customer_ids: List[str],
        ate_ci: Optional[Tuple[float, float]] = None,
        feature_cols: Optional[List[str]] = None,
        X: Optional[np.ndarray] = None,
        action_label: str = "offer_coupon",
    ) -> List[Dict[str, str]]:
		indices = np.argsort(-uplift)
		sorted_ids = [customer_ids[i] for i in indices]
		sorted_uplift = uplift[indices]


		selected: List[int] = []


		# Significance gate using ATE CI: require positive effect with desired confidence
		if ate_ci is not None and self.config.min_ate is not None:
			lo, hi = ate_ci
			if not (lo >= self.config.min_ate):
				# No recommendations if overall effect not significant enough
				return []


        # Budget cap by fraction
		cap = len(sorted_ids)
		if self.config.budget_fraction is not None:
			cap = max(1, int(np.floor(self.config.budget_fraction * len(sorted_ids))))


        for i in range(cap):
            if sorted_uplift[i] >= self.config.min_uplift:
                selected.append(indices[i])


        # Choose action: if catalog present, pick min-cost by default (placeholder for ROI model)
        chosen_action = action_label
        if self.config.actions:
            elig = [a for a in self.config.actions if a.eligible]
            if elig:
                elig_sorted = sorted(elig, key=lambda a: a.cost)
                chosen_action = elig_sorted[0].label

        recs: List[Dict[str, str]] = []
        for idx in selected:
            cid = customer_ids[idx]
            score = float(uplift[idx])
            expl = self._explain_single(idx, X, feature_cols) if X is not None and feature_cols is not None else "uplift above threshold"
            recs.append({
                "customer_id": str(cid),
                "uplift_score": f"{score:.6f}",
                "recommend_action": chosen_action,
                "explanation": expl,
            })
        return recs


	def _explain_single(self, idx: int, X: np.ndarray, feature_cols: List[str]) -> str:
		row = X[idx]
		# Simple heuristic explanation: top absolute features
		abs_vals = np.abs(row)
		order = np.argsort(-abs_vals)[:3]
		pairs = [f"{feature_cols[j]}={row[j]:.3f}" for j in order]
		return "; ".join(pairs)
