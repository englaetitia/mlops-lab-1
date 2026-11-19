"""Quality gate: decide if a candidate model version can become `champion`.

Usage
-----
    uv run python src/food11/promote.py --version 3
    uv run python src/food11/promote.py --version 3 --n 1096   # test split size

Rule (from the lecture):
    p  = test accuracy of the current champion
    se = sqrt(p * (1 - p) / n)        n = number of test images
    if acc_candidate >= acc_champion + 2 * se  -> candidate becomes `champion`
    else                                        -> candidate gets `challenger`

In both cases the decision and the numbers are written as tags on the
candidate version, so the reason is visible in the MLflow UI.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "food11"
METRIC = "test_accuracy"


def accuracy_and_n(client: MlflowClient, version) -> tuple[float, int | None]:
    """Test accuracy of the run that produced this version, and its test size."""
    run = client.get_run(version.run_id)
    if METRIC not in run.data.metrics:
        raise SystemExit(f"run {version.run_id} has no metric '{METRIC}'")
    n = run.data.params.get("test_images")
    return run.data.metrics[METRIC], int(n) if n else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="candidate version number")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--n", type=int, default=None,
                        help="test split size (default: read from the champion run)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    candidate = client.get_model_version(args.model, args.version)
    acc_candidate, n_candidate = accuracy_and_n(client, candidate)

    try:
        champion = client.get_model_version_by_alias(args.model, "champion")
    except MlflowException:
        champion = None

    # No champion yet: nothing to compare with, the candidate becomes champion.
    if champion is None:
        client.set_registered_model_alias(args.model, "champion", candidate.version)
        client.set_model_version_tag(args.model, candidate.version, "gate_decision",
                                     "promoted (no previous champion)")
        client.set_model_version_tag(args.model, candidate.version,
                                     "candidate_accuracy", f"{acc_candidate:.4f}")
        print(f"no champion yet -> v{candidate.version} is now champion "
              f"(accuracy {acc_candidate:.4f})")
        return 0

    if champion.version == candidate.version:
        print(f"v{candidate.version} is already the champion, nothing to do")
        return 0

    acc_champion, n_champion = accuracy_and_n(client, champion)
    n = args.n or n_champion or n_candidate
    if not n:
        raise SystemExit("test split size unknown: pass --n <number of test images>")

    se = math.sqrt(acc_champion * (1 - acc_champion) / n)
    threshold = acc_champion + 2 * se
    promoted = acc_candidate >= threshold

    if promoted:
        client.set_registered_model_alias(args.model, "champion", candidate.version)
        decision = f"promoted to champion (replaces v{champion.version})"
    else:
        client.set_registered_model_alias(args.model, "challenger", candidate.version)
        decision = f"challenger (v{champion.version} stays champion)"

    tags = {
        "gate_decision": decision,
        "candidate_accuracy": f"{acc_candidate:.4f}",
        "champion_accuracy": f"{acc_champion:.4f}",
        "champion_version": str(champion.version),
        "gate_threshold": f"{threshold:.4f}",
        "gate_se": f"{se:.4f}",
        "gate_n": str(n),
    }
    for key, value in tags.items():
        client.set_model_version_tag(args.model, candidate.version, key, value)

    print(f"candidate v{candidate.version}: accuracy {acc_candidate:.4f}")
    print(f"champion  v{champion.version}: accuracy {acc_champion:.4f}  (n={n})")
    print(f"se = {se:.4f}   2*se = {2 * se:.4f}   threshold = {threshold:.4f}")
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
