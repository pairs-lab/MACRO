"""The publishable grid runtime exposes only the current guarded model."""

import io
import unittest
from contextlib import redirect_stdout

from simulations.standalone.app import StandaloneSimulation, _make_release_refiner


class CurrentPredictionModelTests(unittest.TestCase):
    def test_guarded_is_default(self):
        from pathlib import Path
        scenario = Path(__file__).resolve().parents[1] / "maps/map1/scenario_1.json"
        with redirect_stdout(io.StringIO()):
            simulation = StandaloneSimulation(scenario, async_planning=False)
        self.assertEqual(simulation.release_prediction_model, "dynamic_events_guarded")

    def test_removed_models_fail_explicitly(self):
        for model in ("events", "replay", "dynamic_events", "dynamic_events_v1",
                      "dynamic_events_verified"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                _make_release_refiner([], [], {}, {}, frozenset(), .1, 1., 1., set(), model)
