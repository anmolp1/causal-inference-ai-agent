import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd

from causal_agent.data_interface import DataInterface, DatasetSpec
from causal_agent.agent import CausalInferenceAgent, CausalInferenceConfig
from causal_agent.policy import NextBestActionPolicy, PolicyConfig
from causal_agent.service import app as service_app  # expose FastAPI app for uvicorn


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Causal Inference Agent on CSV data")
	parser.add_argument("--customers", type=str, required=False)
	parser.add_argument("--events", type=str, required=False)
	parser.add_argument("--campaigns", type=str, required=False)
	parser.add_argument("--experiments", type=str, required=False)
	parser.add_argument("--id-col", type=str, required=True)
	parser.add_argument("--treatment-col", type=str, required=True)
	parser.add_argument("--outcome-col", type=str, required=True)
	parser.add_argument("--features", type=str, nargs="*", default=None, help="Optional feature columns list")
	parser.add_argument("--output", type=str, required=False, help="Path to write recommendations CSV")
	parser.add_argument("--policy-budget", type=float, required=False, default=None)
	parser.add_argument("--policy-min-uplift", type=float, required=False, default=0.0)
	args = parser.parse_args()

	spec = DatasetSpec(
		customer_id_col=args.id_col,
		treatment_col=args.treatment_col,
		outcome_col=args.outcome_col,
		feature_cols=args.features,
	)
	data = DataInterface(spec)
	df = data.load_from_csv(
		customers_csv=args.customers,
		events_csv=args.events,
		campaigns_csv=args.campaigns,
		experiments_csv=args.experiments,
	)

	X, w, y, feature_cols = data.to_matrices(df)
	customer_ids = df[args.id_col].astype(str).fillna("").tolist()

	agent = CausalInferenceAgent(CausalInferenceConfig())
	est = agent.estimate(X, w, y)

	policy = NextBestActionPolicy(
		PolicyConfig(min_uplift=args.policy_min_uplift, budget_fraction=args.policy_budget)
	)
	recs = policy.recommend(
		uplift=est.uplift,
		customer_ids=customer_ids,
		ate_ci=est.ate_ci,
		feature_cols=feature_cols,
		X=X,
	)

	out_df = pd.DataFrame(recs)
	if args.output:
		out_df.to_csv(args.output, index=False)
	else:
		print(out_df.to_csv(index=False))


if __name__ == "__main__":
	main()
