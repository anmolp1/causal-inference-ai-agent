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

from .agent import CausalInferenceAgent, CausalInferenceConfig
from .policy import NextBestActionPolicy, PolicyConfig, ActionSpec


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


