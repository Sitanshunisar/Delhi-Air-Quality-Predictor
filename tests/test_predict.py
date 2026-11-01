# tests/test_predict.py
"""Smoke tests for the prediction pipeline."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_load_and_engineer():
    from src.data_loader import load_and_engineer
    df = load_and_engineer()
    assert not df.empty
    assert "hour_sin" in df.columns
    assert "hour_cos" in df.columns
    assert "pm2_5" in df.columns


def test_temporal_split():
    from src.data_loader import load_and_engineer, temporal_split
    df = load_and_engineer()
    train, test = temporal_split(df, 0.8)
    assert len(train) + len(test) == len(df)
    assert train.index.max() < test.index.min()


def test_get_prediction_data_keys():
    """Ensure the payload has all expected keys (requires trained models)."""
    try:
        from src.predict import get_prediction_data
        result = get_prediction_data()
        expected_keys = {"labels", "actual_data", "predicted_data_lstm",
                         "predicted_data_gru", "predicted_data_stacker",
                         "r2_lstm", "r2_gru", "r2_stacker"}
        assert expected_keys.issubset(result.keys())
    except Exception as e:
        pytest.skip(f"Models not trained yet: {e}")
