import json

import pandas as pd

from simulation import SimulationConfig
from utils import config_to_json, dataframe_to_csv_bytes


def test_csv_export_round_trip():
    frame = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    payload = dataframe_to_csv_bytes(frame).decode("utf-8")
    assert payload.startswith("a,b")
    assert "1,3.5" in payload


def test_json_export_contains_reproducibility_fields():
    config = SimulationConfig()
    decoded = json.loads(config_to_json(config.to_dict()))
    for key in ["sample_size", "replications", "true_effect", "noise_sd", "estimator", "assignment_mode", "seed"]:
        assert key in decoded
