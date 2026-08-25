"""MLflow helpers using the repository's shared C1/C2 tracking store."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import mlflow
from mlflow.entities import Run

_MLFLOW_TRACKING_URI = "file:./outputs/logs/mlflow"
_MLFLOW_STORE_DIR = Path("outputs/logs/mlflow")


def _normalise_param(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


@contextmanager
def start_c3_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, Any] | None = None,
) -> Iterator[Run]:
    """Start a C3 run in the existing file-backed MLflow store."""

    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a non-empty string")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")

    _MLFLOW_STORE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active_run:
        if params:
            mlflow.log_params(
                {key: _normalise_param(value) for key, value in params.items()}
            )
        yield active_run
