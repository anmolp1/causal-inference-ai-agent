from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
try:
	from neo4j import GraphDatabase  # type: ignore
	neo4j_available = True
except Exception:  # pragma: no cover
	neo4j_available = False


class KnowledgeGraphClient:
	"""Neo4j-backed client for Knowledge Graph access.
	If neo4j driver is unavailable, methods fall back to empty dataframes so the
	rest of the pipeline can run in CSV-only mode.
	"""

	def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
		self.uri = uri
		self.user = user
		self.password = password
		self._driver = GraphDatabase.driver(uri, auth=(user, password)) if (neo4j_available and uri and user and password) else None

	def close(self) -> None:
		if self._driver is not None:
			self._driver.close()

	def fetch_customer_features(self) -> pd.DataFrame:
		if self._driver is None:
			return pd.DataFrame()
		query = (
			"""
			MATCH (c:Customer)
			RETURN c.id AS customer_id, c.segment AS segment, c.churn_score AS churn_score
			"""
		)
		with self._driver.session() as session:
			records = session.run(query)
			rows = [r.data() for r in records]
			return pd.DataFrame(rows)

	def fetch_treatment_outcome(self) -> pd.DataFrame:
		if self._driver is None:
			return pd.DataFrame()
		query = (
			"""
			MATCH (c:Customer)-[t:TARGETED_BY]->(cmp:Campaign)
			OPTIONAL MATCH (c)-[r:RESPONDED_TO]->(cmp)
			RETURN c.id AS customer_id, toInteger(t.assigned) AS treatment, toFloat(COALESCE(r.converted, 0)) AS outcome
			"""
		)
		with self._driver.session() as session:
			records = session.run(query)
			rows = [r.data() for r in records]
			return pd.DataFrame(rows)

	def write_causal_estimates(self, estimates: pd.DataFrame) -> int:
		"""Write per-customer uplift to KG as (:CausalEstimate) nodes linked to Customer."""
		if self._driver is None or estimates.empty:
			return 0
		query = (
			"""
			UNWIND $rows AS row
			MATCH (c:Customer {id: row.customer_id})
			MERGE (e:CausalEstimate {id: row.estimate_id})
			SET e.uplift = row.uplift, e.ate = row.ate, e.ate_lo = row.ate_lo, e.ate_hi = row.ate_hi, e.model_version = row.model_version, e.ts = row.ts
			MERGE (c)-[:HAS_CAUSAL_ESTIMATE]->(e)
			"""
		)
		rows = estimates.to_dict(orient="records")
		with self._driver.session() as session:
			result = session.run(query, rows=rows)
			return result.consume().counters.nodes_created
