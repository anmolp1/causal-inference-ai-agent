## Causal Inference Agent

This agent estimates true cause-effect relationships between marketing actions and customer outcomes. It computes ATE/CATE, customer-level uplift scores, and recommends next-best actions grounded in causal impact rather than correlation.

### Features
- CATE/uplift estimation using doubly-robust/meta-learner approaches (EconML/DoWhy + scikit-learn)
- Targets persuadable customers and avoids wasted spend on sure-things or lost-causes
- Statistical significance checks (bootstrap CIs) and optional CI-based abstain logic
- ROI-aware policy: optimize E[value × uplift − cost] with action costs, total budget, alternatives
- Validation and fairness: hooks to Bias/Validation service with fairness slices
- Observability: Prometheus metrics, health, and evaluation endpoints
- Knowledge Graph (Neo4j) round-trip: configurable Cypher read → estimate → write
- Optional streaming consumer (Kafka) for event-driven scoring
- Model lifecycle: MLflow logging hooks and on-disk persistence

### Inputs
- Historical campaign logs with treatment indicator `treatment` (0/1)
- Outcomes such as `churned` (0/1) or engagement metric
- Customer features (demographics, behavior)
- Optional A/B testing logs

### Quickstart (CSV batch)
```bash
python cli.py \
  --customers /path/customers.csv \
  --events /path/events.csv \
  --campaigns /path/campaigns.csv \
  --experiments /path/experiments.csv \
  --id-col customer_id \
  --treatment-col treatment \
  --outcome-col churned \
  --output /path/recommendations.csv
```

The output CSV includes `customer_id`, `uplift_score`, `recommend_action`, `explanation`.

### Service API (FastAPI)

- GET `/health`: service status
- POST `/estimate`: compute ATE, CI, and per-customer uplift
  - Body: `{features, treatment, outcome, config?, artifacts_path?}`
- POST `/recommend`: get next-best action recommendations
  - Body: `{features, treatment, outcome, customer_ids, policy?, actions?, action_label?, customer_values?, fairness_groups?}`
  - Returns recommendations and optional `validation` summary
- POST `/evaluate`: batch evaluation metrics
  - Body: `{uplift_scores, treatment, outcome, k_fracs?}` → `{metrics:{qini_auc, auuc, uplift_at_10, ...}}`
- GET `/metrics`: Prometheus metrics

Run the service:
```bash
uvicorn causal_agent.service:app --host 0.0.0.0 --port 8000
```

### Client SDK (Python)

Minimal client in `causal_agent/client.py`:
```python
from causal_agent.client import CausalAgentClient
client = CausalAgentClient(base_url="http://localhost:8000")
client.health()
est = client.estimate(features, treatment, outcome)
recs = client.recommend(features, treatment, outcome, customer_ids,
                        policy={"budget_total": 1000.0},
                        actions=[{"action_id":"a1","label":"offer_10","cost":10.0,"eligible":True}],
                        action_label="default")
```

### Knowledge Graph Round-Trip (Neo4j)

POST `/kg/estimate-write` reads customer features and treatment/outcomes via provided Cypher, estimates uplift, and writes per-customer `CausalEstimate` nodes.

Request fields:
- `kg_uri`, `kg_user`, `kg_password`
- `cypher_features`, `cypher_treatment_outcome`
- Mapping: `id_field`, `feature_fields?`, `treatment_field`, `outcome_field`
- Write options: `write_model_version`, `write_estimate_prefix`

### Validation & Fairness

- `recommend` integrates with a Bias/Validation service (config via `VALIDATION_URL`).
- Sends fairness slices: mean uplift by provided group labels (aligned to `customer_ids`).

### Policy (Next Best Action)

- Optimizes expected net value E[value × uplift − cost]
- Supports per-action cost catalog, total budget cap, and optional `customer_values`
- Abstains when ATE CI crosses zero (configurable), returns alternative ranked actions

### Evaluation Utilities

- Qini AUC, AUUC, and uplift@k available via `/evaluate` and `causal_agent/metrics.py`.

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If `econml` or `dowhy` are not available on your platform, the agent will fallback to scikit-learn meta-learners.

### Validation & Experiments
- Run controlled experiments: target top deciles by uplift vs. propensity or random
- Use `/evaluate` for Qini/AUUC/uplift@k; inspect bootstrap CIs for significance
- Integrate with the Bias/Validation service for guardrails and audit artifacts

### Notes
- Ensure treatment assignment is ignorable/has overlap or include appropriate controls and instruments.
- Prefer randomized experiments when possible; for observational data, include confounders and perform sensitivity checks.

### Streaming (Optional)
- Set `KAFKA_BROKERS` and `KAFKA_TOPIC_CUSTOMER_AT_RISK` to enable an event-driven consumer for estimation.

### Observability
- Prometheus metrics: requests, latency, uplift_mean; extendable with dashboards/alerts.

### Security & Privacy (to be configured per deployment)
- API auth (API keys/JWT), RBAC, TLS termination, PII redaction, and feature allowlists are recommended.
