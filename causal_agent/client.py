from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests


class CausalAgentClient:
	"""Lightweight client for the Causal Inference Agent service."""

	def __init__(self, base_url: str, api_key: Optional[str] = None, timeout_seconds: float = 10.0) -> None:
		self.base_url = base_url.rstrip("/")
		self.api_key = api_key
		self.timeout_seconds = timeout_seconds

	def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
		h = {"Content-Type": "application/json"}
		if self.api_key:
			h["X-API-Key"] = self.api_key
		if extra:
			h.update(extra)
		return h

	def health(self) -> Dict[str, Any]:
		r = requests.get(f"{self.base_url}/health", timeout=self.timeout_seconds)
		r.raise_for_status()
		return r.json()

	def estimate(self, features: List[List[float]], treatment: List[int], outcome: List[float], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		payload = {"features": features, "treatment": treatment, "outcome": outcome, "config": config or {}}
		r = requests.post(f"{self.base_url}/estimate", json=payload, headers=self._headers(), timeout=self.timeout_seconds)
		r.raise_for_status()
		return r.json()

	def recommend(
		self,
		features: List[List[float]],
		treatment: List[int],
		outcome: List[float],
		customer_ids: List[str],
		policy: Optional[Dict[str, Any]] = None,
		actions: Optional[List[Dict[str, Any]]] = None,
		action_label: str = "offer_coupon",
		extra_headers: Optional[Dict[str, str]] = None,
	) -> Dict[str, Any]:
		payload: Dict[str, Any] = {
			"features": features,
			"treatment": treatment,
			"outcome": outcome,
			"customer_ids": customer_ids,
			"policy": policy or {},
			"actions": actions,
			"action_label": action_label,
		}
		r = requests.post(f"{self.base_url}/recommend", json=payload, headers=self._headers(extra_headers), timeout=self.timeout_seconds)
		r.raise_for_status()
		return r.json()


