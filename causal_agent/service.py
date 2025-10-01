from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, conlist
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os
import json
from threading import Thread
from kafka import KafkaConsumer
from datetime import datetime

from .agent import CausalInferenceAgent, CausalInferenceConfig
from .policy import NextBestActionPolicy, PolicyConfig, ActionSpec
from .kg_client import KnowledgeGraphClient
from .validation_client import ValidationClient


# Prometheus metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter("causal_agent_requests_total", "Total requests", ["endpoint", "status"], registry=registry)
REQUEST_LATENCY = Histogram("causal_agent_request_latency_seconds", "Request latency", ["endpoint"], registry=registry)
UPLIFT_MEAN = Histogram("causal_agent_uplift_mean", "Mean uplift distribution", buckets=[-1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5], registry=registry)


app = FastAPI(title="Causal Inference Agent Service", version="0.1.0")


class EstimateRequest(BaseModel):
	features: conlist(conlist(float, min_items=1), min_items=1) = Field(..., description="2D array of features [n_samples, n_features]")
	treatment: conlist(int, min_items=1) = Field(..., description="Binary treatment assignments length n_samples")
	outcome: conlist(float, min_items=1) = Field(..., description="Outcome values length n_samples")
	config: Optional[Dict[str, Any]] = Field(default=None, description="Optional agent config overrides")
	artifacts_path: Optional[str] = Field(default=None, description="Optional path to save model artifacts")
	
	class Config:
		extra = "ignore"


class EstimateResponse(BaseModel):
	ate: float
	ate_ci: Optional[Tuple[float, float]]
	uplift: List[float]


class ActionSpecModel(BaseModel):
	action_id: str
	cost: float = 0.0
	label: str = "offer"
	eligible: bool = True


class RecommendRequest(EstimateRequest):
	customer_ids: conlist(str, min_items=1)
	policy: Optional[Dict[str, Any]] = Field(default=None, description="Policy config overrides")
	actions: Optional[List[ActionSpecModel]] = None
	action_label: str = "offer_coupon"
	
	class Config:
		extra = "ignore"

	# Optional fairness grouping inputs: mapping group_name -> list aligned to customer_ids
	fairness_groups: Optional[Dict[str, List[str]]] = None
	# Optional customer value vector aligned to customer_ids for ROI scoring
	customer_values: Optional[List[float]] = None


class Recommendation(BaseModel):
	customer_id: str
	uplift_score: str
	recommend_action: str
	explanation: str


class RecommendResponse(BaseModel):
	recommendations: List[Recommendation]
	validation: Optional[Dict[str, Any]] = None


def _build_agent(cfg_overrides: Optional[Dict[str, Any]]) -> CausalInferenceAgent:
	cfg = CausalInferenceConfig(**cfg_overrides) if cfg_overrides else CausalInferenceConfig()
	return CausalInferenceAgent(cfg)


def _build_policy(cfg_overrides: Optional[Dict[str, Any]]) -> NextBestActionPolicy:
	cfg = PolicyConfig(**cfg_overrides) if cfg_overrides else PolicyConfig()
	return NextBestActionPolicy(cfg)


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/estimate", response_model=EstimateResponse)
def estimate(req: EstimateRequest, request: Request) -> EstimateResponse:
	endpoint = "/estimate"
	with REQUEST_LATENCY.labels(endpoint).time():
		try:
			agent = _build_agent(req.config)
			X = np.asarray(req.features, dtype=float)
			w = np.asarray(req.treatment, dtype=int)
			y = np.asarray(req.outcome, dtype=float)
			est = agent.estimate(X, w, y)
			# Optional save of artifacts
			if req.artifacts_path:
				try:
					agent.save(req.artifacts_path)
				except Exception:
					pass
			REQUEST_COUNT.labels(endpoint, "200").inc()
			try:
				UPLIFT_MEAN.observe(float(np.mean(est.uplift)))
			except Exception:
				pass
			return EstimateResponse(ate=est.ate, ate_ci=est.ate_ci, uplift=est.uplift.tolist())
		except Exception as e:
			REQUEST_COUNT.labels(endpoint, "500").inc()
			raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request) -> RecommendResponse:
	endpoint = "/recommend"
	with REQUEST_LATENCY.labels(endpoint).time():
		try:
			# Reuse estimate logic to get uplift
			agent = _build_agent(req.config)
			X = np.asarray(req.features, dtype=float)
			w = np.asarray(req.treatment, dtype=int)
			y = np.asarray(req.outcome, dtype=float)
			est = agent.estimate(X, w, y)

			# Map incoming actions into PolicyConfig if provided
			pcfg: Dict[str, Any] = dict(req.policy or {})
			if req.actions is not None:
				pcfg["actions"] = [ActionSpec(action_id=a.action_id, cost=a.cost, label=a.label, eligible=a.eligible) for a in req.actions]
			policy = _build_policy(pcfg)
			recs = policy.recommend(
				uplift=np.asarray(est.uplift),
				customer_ids=list(req.customer_ids),
				ate_ci=est.ate_ci,
				feature_cols=None,
				X=None,
				action_label=req.action_label,
				customer_values=np.asarray(req.customer_values, dtype=float) if req.customer_values is not None else None,
			)

			# Build fairness slices if provided
			fairness: Dict[str, Dict[str, Dict[str, float]]] = {}
			if req.fairness_groups:
				upl = np.asarray(est.uplift)
				for gname, values in req.fairness_groups.items():
					if not values or len(values) != len(req.customer_ids):
						continue
					group_map: Dict[str, Dict[str, float]] = {}
					# aggregate by value
					stats: Dict[str, List[float]] = {}
					for i, val in enumerate(values):
						stats.setdefault(str(val), []).append(float(upl[i]))
					for val, arr in stats.items():
						if arr:
							mean_u = float(np.mean(arr))
							group_map[str(val)] = {"mean_uplift": mean_u, "count": float(len(arr))}
					fairness[gname] = group_map

			# Validation hook (real client, env-configurable)
			validator = ValidationClient()
			validation_payload = {
				"ate": est.ate,
				"ate_ci": est.ate_ci,
				"recommendations": recs,
				"fairness_slices": fairness,
			}
			validation_result = validator.validate(validation_payload)
			REQUEST_COUNT.labels(endpoint, "200").inc()
			return RecommendResponse(recommendations=[Recommendation(**r) for r in recs], validation=validation_result)
		except Exception as e:
			REQUEST_COUNT.labels(endpoint, "500").inc()
			raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics() -> Response:
	data = generate_latest(registry)
	return Response(content=data, media_type=CONTENT_TYPE_LATEST)


class KGRoundTripRequest(BaseModel):
	kg_uri: str
	kg_user: str
	kg_password: str
	cypher_features: str
	cypher_treatment_outcome: str
	write_model_version: str = "v0"
	write_estimate_prefix: str = "causal_estimate"
	# Schema mapping
	id_field: str = "customer_id"
	feature_fields: Optional[List[str]] = None
	treatment_field: str = "treatment"
	outcome_field: str = "outcome"


@app.post("/kg/estimate-write")
def kg_estimate_write(req: KGRoundTripRequest) -> Dict[str, int]:
	endpoint = "/kg/estimate-write"
	with REQUEST_LATENCY.labels(endpoint).time():
		try:
			kg = KnowledgeGraphClient(uri=req.kg_uri, user=req.kg_user, password=req.kg_password)
			try:
				with kg._driver.session() as session:  # type: ignore[attr-defined]
					feat_rows = [r.data() for r in session.run(req.cypher_features)]
					to_rows = [r.data() for r in session.run(req.cypher_treatment_outcome)]
			except Exception:
				# If KG not available, treat as zero work
				return {"written": 0}

			# Join on configured id field in-memory
			feat_ids = {str(r.get(req.id_field)): r for r in feat_rows}
			X: List[List[float]] = []
			w: List[int] = []
			y: List[float] = []
			cust_ids: List[str] = []
			for r in to_rows:
				cid = str(r.get(req.id_field))
				if cid not in feat_ids:
					continue
				f = feat_ids[cid]
				# Build feature vector from configured fields or all numeric except id
				fv: List[float] = []
				if req.feature_fields is not None:
					for k in req.feature_fields:
						if k == req.id_field:
							continue
						try:
							fv.append(float(f.get(k, 0)))
						except Exception:
							fv.append(0.0)
				else:
					for k, v in f.items():
						if k == req.id_field:
							continue
						try:
							fv.append(float(v))
						except Exception:
							continue
				X.append(fv)
				w.append(int(r.get(req.treatment_field, 0)))
				y.append(float(r.get(req.outcome_field, 0.0)))
				cust_ids.append(cid)

			if not X:
				return {"written": 0}

			agent = _build_agent({"random_state": 42})
			est = agent.estimate(np.asarray(X, dtype=float), np.asarray(w, dtype=int), np.asarray(y, dtype=float))

			# Build rows for write back
			ts = datetime.utcnow().isoformat()
			rows = []
			for i, cid in enumerate(cust_ids):
				rows.append({
					"customer_id": cid,
					"estimate_id": f"{req.write_estimate_prefix}:{cid}:{ts}",
					"uplift": float(est.uplift[i]),
					"ate": float(est.ate),
					"ate_lo": float(est.ate_ci[0]) if est.ate_ci else None,
					"ate_hi": float(est.ate_ci[1]) if est.ate_ci else None,
					"model_version": req.write_model_version,
					"ts": ts,
				})

			import pandas as pd  # local import to avoid heavy dep on import time
			df = pd.DataFrame(rows)
			written = kg.write_causal_estimates(df)
			REQUEST_COUNT.labels(endpoint, "200").inc()
			return {"written": int(written)}
		except Exception as e:
			REQUEST_COUNT.labels(endpoint, "500").inc()
			raise HTTPException(status_code=500, detail=str(e))


# Optional Kafka consumer for event-driven scoring (runs if env vars set)
def _start_kafka_consumer() -> None:
	brokers = os.getenv("KAFKA_BROKERS")
	topic = os.getenv("KAFKA_TOPIC_CUSTOMER_AT_RISK")
	if not brokers or not topic:
		return
	consumer = KafkaConsumer(
		topic,
		bootstrap_servers=brokers.split(","),
		value_deserializer=lambda m: json.loads(m.decode("utf-8")),
		auto_offset_reset="latest",
		enable_auto_commit=True,
	)
	for msg in consumer:
		try:
			payload = msg.value
			# Expect payload with keys: features, treatment, outcome, customer_ids
			agent = _build_agent(None)
			X = np.asarray(payload.get("features", []), dtype=float)
			w = np.asarray(payload.get("treatment", []), dtype=int)
			y = np.asarray(payload.get("outcome", []), dtype=float)
			if X.size == 0:
				continue
			est = agent.estimate(X, w, y)
			_ = np.mean(est.uplift)  # touch to ensure compute, metrics updated in API only
		except Exception:
			continue


def _maybe_start_background_threads() -> None:
	try:
		Thread(target=_start_kafka_consumer, daemon=True).start()
	except Exception:
		pass


_maybe_start_background_threads()


