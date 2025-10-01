from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetSpec:
	customer_id_col: str
	treatment_col: str
	outcome_col: str
	feature_cols: Optional[List[str]] = None


class DataInterface:
	"""Data preparation utilities for causal modeling.
	Accepts CSVs or pulls from a Knowledge Graph. Produces aligned matrices
	for features X, treatment w, and outcome y.
	"""

	def __init__(self, spec: DatasetSpec) -> None:
		self.spec = spec

	def load_from_csv(
		self,
		customers_csv: Optional[str] = None,
		events_csv: Optional[str] = None,
		campaigns_csv: Optional[str] = None,
		experiments_csv: Optional[str] = None,
	) -> pd.DataFrame:
		frames: List[pd.DataFrame] = []
		if customers_csv:
			frames.append(pd.read_csv(customers_csv))
		if events_csv:
			frames.append(pd.read_csv(events_csv))
		if campaigns_csv:
			frames.append(pd.read_csv(campaigns_csv))
		if experiments_csv:
			frames.append(pd.read_csv(experiments_csv))

		if not frames:
			raise ValueError("No CSV inputs provided.")

		# Perform a series of outer joins on customer_id to aggregate features.
		merged = frames[0]
		for df in frames[1:]:
			key = self.spec.customer_id_col if self.spec.customer_id_col in merged.columns and self.spec.customer_id_col in df.columns else None
			if key is not None:
				merged = merged.merge(df, on=key, how="outer")
			else:
				merged = pd.concat([merged, df], axis=1)

		return merged

	def to_matrices(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
		missing = [c for c in [self.spec.treatment_col, self.spec.outcome_col] if c not in df.columns]
		if missing:
			raise ValueError(f"Missing required columns: {missing}")

		if self.spec.feature_cols is None:
			feature_cols = [
				c for c in df.columns
				if c not in {self.spec.customer_id_col, self.spec.treatment_col, self.spec.outcome_col}
			]
		else:
			feature_cols = self.spec.feature_cols

		X = df[feature_cols].select_dtypes(include=[np.number, "bool"]).fillna(0.0).astype(float).values
		w = df[self.spec.treatment_col].astype(int).values
		y = df[self.spec.outcome_col].astype(float).values
		return X, w, y, feature_cols
