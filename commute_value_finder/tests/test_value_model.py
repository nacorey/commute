import numpy as np
import pandas as pd
from src.preprocessor import preprocess
from src.value_model import fit_quality_model


def test_fit_quality_model_returns_residuals(sample_transactions):
    clean = preprocess(sample_transactions)
    model, scored, r2 = fit_quality_model(clean)
    assert "resid" in scored.columns
    assert len(scored) == len(clean)
    assert abs(scored["resid"].mean()) < 1e-6
    assert 0.0 <= r2 <= 1.0
