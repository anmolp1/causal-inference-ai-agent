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
    budget_total: Optional[float] = None  # absolute budget cap across selections
    require_positive_net_value: bool = True  # abstain if expected net <= 0
    abstain_if_ci_crosses_zero: bool = True  # abstain if ATE CI spans <= 0
    return_alternatives: bool = True
    num_alternatives: int = 2




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
        customer_values: Optional[np.ndarray] = None,
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


        # ROI-aware selection: compute expected net value per customer and pick best action
        values = customer_values if customer_values is not None else np.ones_like(uplift, dtype=float)

        candidates: List[Tuple[int, str, float, float]] = []  # (idx, action_label, cost, net_value)
        for idx in selected:
            u = float(uplift[idx])
            v = float(values[idx])
            # CI-based abstain check (using ATE CI as proxy)
            if self.config.abstain_if_ci_crosses_zero and ate_ci is not None:
                lo, hi = ate_ci
                if lo <= 0 <= hi:
                    # skip this customer entirely if uncertain overall effect
                    continue

            best_label = action_label
            best_cost = 0.0
            best_net = u * v
            ranked: List[Tuple[str, float, float]] = []  # (label, cost, net)
            if self.config.actions:
                elig = [a for a in self.config.actions if a.eligible]
                if elig:
                    best_net = -1e18
                    for a in elig:
                        net = u * v - float(a.cost)
                        if net > best_net:
                            best_net = net
                            best_label = a.label
                            best_cost = float(a.cost)
                        ranked.append((a.label, float(a.cost), net))
            else:
                # If no actions provided, assume zero cost for the default action
                best_cost = 0.0
                best_net = u * v
                ranked.append((best_label, best_cost, best_net))

            if (not self.config.require_positive_net_value) or (best_net > 0):
                candidates.append((idx, best_label, best_cost, best_net))
            # attach ranked alternatives to object cache for later explanation
            # We will recompute lightweight alternatives when building recs

        # Sort by net value descending
        candidates.sort(key=lambda t: -t[3])

        # Apply absolute budget if configured
        total_cost = 0.0
        rec_indices: List[Tuple[int, str, float, float]] = []
        for idx, label, cost, net in candidates[:cap]:
            if self.config.budget_total is not None and (total_cost + cost) > self.config.budget_total:
                continue
            total_cost += cost
            rec_indices.append((idx, label, cost, net))

        recs: List[Dict[str, str]] = []
        for idx, label, cost, net in rec_indices:
            cid = customer_ids[idx]
            score = float(uplift[idx])
            expl = self._explain_single(idx, X, feature_cols) if X is not None and feature_cols is not None else "roi-optimized action"
            rec: Dict[str, str] = {
                "customer_id": str(cid),
                "uplift_score": f"{score:.6f}",
                "recommend_action": label,
                "explanation": expl,
                "expected_net_value": f"{net:.6f}",
            }
            # compute alternatives if enabled
            if self.config.return_alternatives and self.config.actions:
                u = float(uplift[idx])
                v = float(values[idx])
                alts = []
                for a in self.config.actions:
                    if not a.eligible or a.label == label:
                        continue
                    net_alt = u * v - float(a.cost)
                    alts.append({"action": a.label, "expected_net_value": f"{net_alt:.6f}"})
                alts.sort(key=lambda x: -float(x["expected_net_value"]))
                rec["alternative_actions"] = alts[: max(0, int(self.config.num_alternatives))]
            recs.append(rec)
        return recs


	def _explain_single(self, idx: int, X: np.ndarray, feature_cols: List[str]) -> str:
		row = X[idx]
		# Simple heuristic explanation: top absolute features
		abs_vals = np.abs(row)
		order = np.argsort(-abs_vals)[:3]
		pairs = [f"{feature_cols[j]}={row[j]:.3f}" for j in order]
		return "; ".join(pairs)
