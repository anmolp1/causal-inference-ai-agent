from __future__ import annotations

from typing import Any, Dict, Optional
import os
import requests


class ValidationClient:
	"""HTTP client for Bias/Validation service.
	If VALIDATION_URL is not configured, approves by default.
	"""

	def __init__(self, base_url: Optional[str] = None, timeout_seconds: float = 5.0) -> None:
		self.base_url = base_url or os.getenv("VALIDATION_URL")
		self.timeout_seconds = timeout_seconds

	def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
		if not self.base_url:
			return {"approved": True, "notes": "no validation url configured"}
		try:
			r = requests.post(self.base_url.rstrip("/"), json=payload, timeout=self.timeout_seconds)
			r.raise_for_status()
			return r.json()
		except Exception as e:
			return {"approved": False, "notes": f"validation error: {e}"}


