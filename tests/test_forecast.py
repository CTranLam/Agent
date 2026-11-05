import numpy as np, pytest
from diabetes_mas.dl.predictor import Predictor
from diabetes_mas.dl.synthetic import simulate_cgm
from diabetes_mas.dl.train_lstm import forecast_metrics, split_by_series, windows
from tests.fakes import FakePredictor


def test_split_by_series_no_overlap():
    tr, va, te = split_by_series(simulate_cgm(20, seed=1))
    assert not {a.tobytes() for a in tr} & {a.tobytes() for a in te}

def test_windows_shape():
    X, Y = windows(simulate_cgm(2, length=30, seed=1))
    assert X.shape[1:] == (12, 1) and Y.shape[1:] == (6,)

def test_simulator_has_hypo_events():
    assert (simulate_cgm(20, seed=1) < 70).any()

def test_metrics_hold_last_misses_falling_hypo():
    y = np.array([[80, 75, 70, 65, 60, 55.]]); hold = np.full((1, 6), 85.)
    assert forecast_metrics(y, hold)["hypo_recall"] == 0.0
    assert forecast_metrics(y, y)["hypo_recall"] == 1.0

def test_single_reading_low_confidence():  # Review Focus 2
    f = FakePredictor().forecast([120.0])
    assert f["low_confidence"] and f["trend_per_5min"] == 0 and f["n_input"] == 1

def test_predictor_missing_artifact_message(tmp_path):
    with pytest.raises(RuntimeError, match="train"):
        Predictor(tmp_path)
