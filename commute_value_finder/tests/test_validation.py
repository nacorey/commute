import numpy as np
import pandas as pd
from src.validation import backtest


def _make_clean(months, complexes):
    rows = []
    for ym in months:
        for apt, ppsqm in complexes:
            rows.append({
                "구": "A", "법정동": "x", "아파트명": apt,
                "전용면적": 84.0, "층": 5, "age": 10,
                "area_band": "60-85", "거래년월": ym,
                "price_per_sqm": ppsqm, "거래금액": ppsqm * 84,
            })
    return pd.DataFrame(rows)


def test_backtest_splits_and_returns_zone_stats():
    months = [202509, 202510, 202511, 202512, 202601, 202602, 202603, 202604, 202605, 202606]
    clean = _make_clean(months, [("P", 1000.0), ("Q", 1500.0), ("R", 2000.0)])
    commute = pd.DataFrame({"구": ["A"], "법정동": ["x"], "commute_minutes": [30]})
    out = backtest(clean, commute, train_months=7, sigma_mult=1.0,
                   min_tx=1, recency_months=6, shrinkage_k=5)
    assert out["n_test"] == 9     # 마지막 3개월(202604~202606) × 3단지
    assert out["n_train"] == 21
    for z in ["Blue", "Gray", "Red"]:
        assert z in out
