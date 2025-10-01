from __future__ import annotations

import json
from pathlib import Path
from causal_agent.service import app


def main() -> None:
	openapi = app.openapi()
	out = Path("openapi.json")
	out.write_text(json.dumps(openapi, indent=2))
	print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
	main()


