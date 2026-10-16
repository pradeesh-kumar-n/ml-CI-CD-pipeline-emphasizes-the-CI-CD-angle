# ml-CI-CD-pipeline-emphasizes-the-CI-CD-angle

Reproducible MLOps reference implementation demonstrating:

- Deterministic synthetic dataset generation
- Training and tracking model candidates with MLflow
- Comparing experiment runs and promoting the best model to the model registry
- A lightweight CI workflow for tests

This repository packages the example pipeline as a small Python module and includes unit tests and a GitHub Actions workflow.

Project structure

- src/mlops_pipeline/ - pipeline implementation (train, compare, register)
- tests/ - unit tests (pytest)
- .github/workflows/ci.yml - CI test workflow
- pyproject.toml / requirements.txt - packaging and dependencies

Quickstart

1. Create a virtual environment and install dependencies:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # PowerShell on Windows
   python -m pip install -r requirements.txt
   python -m pip install -e .

2. Run the pipeline (uses `sqlite:///mlflow.db` in the current directory by default):

   python -m mlops_pipeline

   To use a different database on Windows PowerShell:

   $trackingUri = "sqlite:///$($PWD.Path.Replace('\', '/'))/mlflow_metrics.db"
   python -m mlops_pipeline --tracking-uri $trackingUri --skip-registration

3. Run the unit tests:

   python -m pytest

Notes

- MLflow: the default SQLite database is created in the current working directory. Override it with `--tracking-uri` to use a different local database or tracking server.
- Model serialization: models are saved in cloudpickle format. Like pickle, cloudpickle can execute code when loading an untrusted model; only load artifacts from trusted sources.
- CI: the included GitHub Actions workflow runs the test suite on pushes and PRs.

Measured output metrics (example run)

The following metrics were produced by a local execution of the pipeline (three candidate runs). These are the exact values printed by the pipeline during that run.

| Run prefix | Parameters                      | Accuracy | Precision | Recall | F1    |
|------------|---------------------------------|----------:|----------:|-------:|------:|
| a2c20b7f   | n_estimators=50, max_depth=5    | 0.9700   | 0.9700   | 0.9700 | 0.9700 |
| 7b52ec55   | n_estimators=100, max_depth=10  | 0.9900   | 0.9902   | 0.9900 | 0.9900 |
| bbdb7707   | n_estimators=200, max_depth=20  | 0.9900   | 0.9902   | 0.9900 | 0.9900 |

Observations

- Two larger ensembles (100 and 200 trees) reached identical accuracy/F1 on this synthetic dataset (0.99).
- The smaller forest (50 trees) performed slightly worse (0.97 accuracy).

If you want the run artifacts and MLflow UI locally, set a tracking URI and then run `mlflow ui --backend-store-uri sqlite:///path/to/mlflow.db` and open http://127.0.0.1:5000.

Contact / License

This example is MIT-licensed; see the LICENSE file for details.
