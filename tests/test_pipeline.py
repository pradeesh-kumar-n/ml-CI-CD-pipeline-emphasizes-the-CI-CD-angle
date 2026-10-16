from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlops_pipeline import MLOpsPipeline, create_ci_cd_workflow


class MLOpsPipelineTests(unittest.TestCase):
    def test_prepare_data_is_reproducible_and_stratified(self) -> None:
        first = MLOpsPipeline.prepare_data()
        second = MLOpsPipeline.prepare_data()

        for first_part, second_part in zip(first, second):
            np.testing.assert_array_equal(first_part, second_part)
        self.assertEqual(first[0].shape, (800, 10))
        self.assertEqual(first[1].shape, (200, 10))
        self.assertEqual(set(first[2]), {0, 1})
        self.assertEqual(set(first[3]), {0, 1})

    def test_train_and_compare_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tracking_dir:
            pipeline = MLOpsPipeline(
                experiment_name="pipeline-test",
                tracking_uri=f"sqlite:///{(Path(tracking_dir) / 'mlflow.db').as_posix()}",
            )
            metrics = pipeline.train_model(
                {"n_estimators": 10, "max_depth": 5}
            )

            self.assertEqual(set(metrics), {"accuracy", "precision", "recall", "f1"})
            self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics.values()))
            runs = pipeline.compare_experiments()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].data.metrics["accuracy"], metrics["accuracy"])

    def test_ci_cd_workflow_has_five_stages(self) -> None:
        workflow = create_ci_cd_workflow()
        self.assertEqual(len(workflow["stages"]), 5)
        self.assertEqual(
            [stage["name"] for stage in workflow["stages"]][0],
            "data_preparation",
        )


if __name__ == "__main__":
    unittest.main()
