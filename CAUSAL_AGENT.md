## Causal Inference Agent

This agent estimates true cause-effect relationships between marketing actions and customer outcomes. It computes ATE/CATE, customer-level uplift scores, and recommends next-best actions grounded in causal impact rather than correlation.

### Features
- CATE/uplift estimation using doubly-robust/meta-learner approaches (EconML/DoWhy + scikit-learn)
- Targets persuadable customers and avoids wasted spend on sure-things or lost-causes
- Statistical significance checks (bootstrap CIs, Wald-style Z tests)
- Explanations for recommendations (key features and segments driving uplift)
- Pluggable data interface (CSV or Knowledge Graph stubs)

### Inputs
- Historical campaign logs with treatment indicator `treatment` (0/1)
- Outcomes such as `churned` (0/1) or engagement metric
- Customer features (demographics, behavior)
- Optional A/B testing logs

### Quickstart (CSV)
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

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If `econml` or `dowhy` are not available on your platform, the agent will fallback to scikit-learn meta-learners.

### Validation
- Run controlled experiments: target top deciles by uplift vs. propensity or random.
- Measure retention/engagement improvement X% vs. non-uplift targeting.
- Inspect bootstrap CIs to ensure effects are statistically significant.

### Notes
- Ensure treatment assignment is ignorable/has overlap or include appropriate controls and instruments.
- Prefer randomized experiments when possible; for observational data, include confounders and perform sensitivity checks.
