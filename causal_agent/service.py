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
			)

			# Placeholder validation hook (to be replaced by real Bias/Validation service)
			validation_result = {"approved": True, "notes": "pass-through"}
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

			# Join on customer_id in-memory
			feat_ids = {r.get("customer_id"): r for r in feat_rows}
			X: List[List[float]] = []
			w: List[int] = []
			y: List[float] = []
			cust_ids: List[str] = []
			for r in to_rows:
				cid = str(r.get("customer_id"))
				if cid not in feat_ids:
					continue
				f = feat_ids[cid]
				# naive feature vector: all numeric values in feature row except id
				fv: List[float] = []
				for k, v in f.items():
					if k == "customer_id":
						continue
					try:
						fv.append(float(v))
					except Exception:
						continue
				X.append(fv)
				w.append(int(r.get("treatment", 0)))
				y.append(float(r.get("outcome", 0.0)))
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


