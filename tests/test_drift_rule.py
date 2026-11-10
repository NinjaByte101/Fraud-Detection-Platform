import json
import sys

import numpy as np
import pandas as pd

from src.monitoring import drift


def test_flag_is_written_when_drift_is_high(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drift.maybe_retrain(0.5, "demo", ["unused"], run=False)
    flag = json.loads((tmp_path / "data" / "retrain_flag.json").read_text())
    assert flag["dataset"] == "demo"
    assert flag["drift_share"] == 0.5


def test_no_flag_when_drift_is_low(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drift.maybe_retrain(0.2, "demo", ["unused"], run=False)
    assert not (tmp_path / "data" / "retrain_flag.json").exists()


def test_threshold_value_itself_does_not_trigger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drift.maybe_retrain(drift.SHARE_THRESHOLD, "demo", ["unused"], run=False)
    assert not (tmp_path / "data" / "retrain_flag.json").exists()


def test_training_command_runs_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = [sys.executable, "-c", "open('ran.txt', 'w').write('ok')"]
    drift.maybe_retrain(0.9, "demo", command, run=False)
    assert not (tmp_path / "ran.txt").exists()
    drift.maybe_retrain(0.9, "demo", command, run=True)
    assert (tmp_path / "ran.txt").exists()


def test_drift_check_finds_the_shifted_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({c: rng.normal(0, 1, 2000) for c in ["a", "b", "c"]})
    current = reference.copy()
    current["a"] = rng.normal(3, 1, 2000)          # only this column moves
    current["b"] = rng.normal(0, 1, 2000)
    current["c"] = rng.normal(0, 1, 2000)

    share = drift.drift_check(reference, current, "synthetic")

    assert abs(share - 1 / 3) < 0.01
    assert (tmp_path / "reports" / "drift_synthetic.html").exists()
