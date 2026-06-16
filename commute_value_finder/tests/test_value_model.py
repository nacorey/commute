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


from src.value_model import aggregate_to_complex


def test_aggregate_to_complex_shrinks_small_samples():
    # 단지 X: 거래 1건, 잔차 +1.0 (동 평균 쪽으로 끌려가야 함)
    scored = pd.DataFrame(
        {
            "구": ["강남구", "강남구", "강남구"],
            "법정동": ["삼성동", "삼성동", "삼성동"],
            "아파트명": ["X", "Y", "Y"],
            "거래년월": [202601, 202602, 202603],
            "resid": [1.0, 0.0, 0.0],
        }
    )
    comp = aggregate_to_complex(scored, k=5)
    x_row = comp[comp["아파트명"] == "X"].iloc[0]
    assert x_row["n"] == 1
    assert x_row["입지가치지수"] < 1.0
    assert x_row["last_ym"] == 202601
