"""Regenerate reports/results_summary.json from local reports (reports/eval is git-ignored)."""

import json
import sys

from rehabpose.ml.results import summarize
from rehabpose.settings import project_root

COMMIT = "b48c5a1"
ABLATIONS = "reports/ablations/ablations-20261101-165706.json"

if __name__ == "__main__":
    root = project_root()
    commit = sys.argv[1] if len(sys.argv) > 1 else COMMIT
    summary = summarize(root, commit, root / ABLATIONS)
    out = root / "reports" / "results_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out, len(summary["configs"]), "configs")
