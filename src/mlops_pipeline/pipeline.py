"""Train, evaluate, compare, and register a Random Forest with MLflow."""

from __future__ import annotations

import argparse
from typing import Any

import mlflow
import numpy as np
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


class MLOpsPipeline:
    """Manage reproducible model runs in one MLflow experiment."""

    def __init__(
        self,
        experiment_name: str = "Multi_AI_Training",
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri or "sqlite:///mlflow.db")
        self.mlflow_client = MlflowClient()
        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def prepare_data(seed: int = 42) -> tuple[np.ndarray, ...]:
        """Generate a deterministic binary classification dataset."""
        features = np.random.default_rng(seed).standard_normal((1000, 10))
        labels = (features[:, 0] + features[:, 1] > 0).astype(int)
        return train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )

    def train_model(self, params: dict[str, Any]) -> dict[str, float]:
        """Fit and log one model run, returning its evaluation metrics."""
        model_params = {"random_state": 42, "n_jobs": 1, **params}
        with mlflow.start_run() as run:
            mlflow.log_params(model_params)
            x_train, x_test, y_train, y_test = self.prepare_data()

            model = RandomForestClassifier(**model_params)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

            print(
                f"Run {run.info.run_id[:8]}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
            )
            return metrics

    def compare_experiments(self) -> list[Run]:
        """Print and return this experiment's runs, best accuracy first."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise RuntimeError(f"MLflow experiment not found: {self.experiment_name}")

        runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"],
        )
        if not runs:
            print("No runs found for this experiment.")
            return []

        for run in runs:
            accuracy = run.data.metrics.get("accuracy", 0.0)
            f1 = run.data.metrics.get("f1", 0.0)
            print(
                f"Run ID: {run.info.run_id[:8]} | "
                f"params={run.data.params} | "
                f"accuracy={accuracy:.4f} | f1={f1:.4f}"
            )
        return runs

    def register_best_model(
        self, model_name: str = "mlops-random-forest-classifier"
    ) -> str:
        """Register the model artifact from the highest-accuracy run."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise RuntimeError(f"MLflow experiment not found: {self.experiment_name}")

        best_runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=1,
        )
        if not best_runs:
            raise RuntimeError(
                f"No completed runs to register in {self.experiment_name!r}"
            )

        best_run = best_runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        print(
            f"Registered {model_name} version {registered_model.version} from "
            f"run {best_run.info.run_id[:8]} "
            f"(accuracy={best_run.data.metrics['accuracy']:.4f})."
        )
        return registered_model.version


def create_ci_cd_workflow() -> dict[str, list[dict[str, Any]]]:
    """Describe the main stages in a model CI/CD lifecycle."""
    workflow = {
        "stages": [
            {
                "name": "data_preparation",
                "description": "Validate input data quality and ranges.",
                "actions": ["Check schema", "Check for nulls", "Validate ranges"],
            },
            {
                "name": "model_training",
                "description": "Train and track versioned model candidates.",
                "actions": ["Log parameters", "Calculate metrics", "Save artifacts"],
            },
            {
                "name": "model_evaluation",
                "description": "Evaluate candidate models against benchmarks.",
                "actions": ["Track metrics", "Compare to baseline", "Check threshold"],
            },
            {
                "name": "model_deployment",
                "description": "Promote an approved model to a serving environment.",
                "actions": ["Register model", "Deploy to endpoint", "Run smoke tests"],
            },
            {
                "name": "model_monitoring",
                "description": "Monitor deployed model quality and data drift.",
                "actions": ["Track predictions", "Detect drift", "Alert on changes"],
            },
        ]
    }
    for stage in workflow["stages"]:
        print(f"{stage['name'].upper()}: {stage['description']}")
        print(f"  Actions: {', '.join(stage['actions'])}")
    return workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train Random Forest candidates and track them with MLflow."
    )
    parser.add_argument(
        "--experiment-name",
        default="Multi_AI_Training",
        help="MLflow experiment name (default: %(default)s)",
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI; defaults to MLflow's local ./mlruns store.",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Do not register the highest-accuracy model.",
    )
    args = parser.parse_args(argv)

    pipeline = MLOpsPipeline(args.experiment_name, args.tracking_uri)
    param_sets = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20},
    ]
    print("Training model candidates...")
    for params in param_sets:
        print(f"\nParameters: {params}")
        pipeline.train_model(params)

    print("\nExperiment comparison (best accuracy first)...")
    pipeline.compare_experiments()

    if not args.skip_registration:
        print("\nRegistering best model...")
        pipeline.register_best_model()

    print("\nCI/CD workflow stages...")
    create_ci_cd_workflow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
