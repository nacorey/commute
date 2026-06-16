"""저평가 신호의 자기검증: 시간 홀드아웃 백테스트 + 공간 자기상관(Moran's I)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from src.value_model import aggregate_to_complex, classify_zones, ym_subtract_months


def _unit_design(df: pd.DataFrame) -> pd.DataFrame:
    """시점 더미 없는 단위특성 설계행렬(백테스트용 — 미래 기간 예측 가능)."""
    X = pd.concat(
        [
            np.log(df["전용면적"]).rename("log_area"),
            df["층"].astype(float).rename("floor"),
            df["age"].astype(float).rename("age"),
            (df["age"].astype(float) ** 2).rename("age2"),
            pd.get_dummies(df["area_band"], prefix="band", drop_first=True),
        ],
        axis=1,
    )
    return X.astype(float)


def backtest(clean: pd.DataFrame, commute: pd.DataFrame, *, train_months,
             sigma_mult, min_tx, recency_months, shrinkage_k) -> dict:
    """앞 기간으로 zone 산출 → 뒤 3개월 실현가로 신호 유효성 검증.

    Blue(저평가 후보)의 이후 실현 잔차가 Gray/Red보다 높으면 신호가 유효.
    """
    latest = int(clean["거래년월"].max())
    test_cut = ym_subtract_months(latest, 3)  # 이보다 큰 거래년월 = test
    train = clean[clean["거래년월"] <= test_cut].copy()
    test = clean[clean["거래년월"] > test_cut].copy()
    if train.empty or test.empty:
        return {"error": "insufficient split", "n_train": len(train), "n_test": len(test)}

    Xtr = _unit_design(train)
    ytr = np.log(train["price_per_sqm"])
    model = LinearRegression().fit(Xtr, ytr)
    train = train.copy()
    train["resid"] = ytr - model.predict(Xtr)

    comp = aggregate_to_complex(train, k=shrinkage_k)
    zones = classify_zones(comp, commute, sigma_mult, min_tx, recency_months,
                           int(train["거래년월"].max()))
    zone_map = zones.set_index(["구", "법정동", "아파트명"])["zone"].to_dict()

    Xte = _unit_design(test).reindex(columns=Xtr.columns, fill_value=0)
    test = test.copy()
    test["realized_resid"] = np.log(test["price_per_sqm"]) - model.predict(Xte)
    test["zone"] = [
        zone_map.get((g, d, a), "Unknown")
        for g, d, a in zip(test["구"], test["법정동"], test["아파트명"])
    ]

    out = {"n_train": len(train), "n_test": len(test)}
    for z in ["Blue", "Gray", "Red"]:
        sub = test[test["zone"] == z]
        out[z] = {
            "n": int(len(sub)),
            "mean_realized_resid": float(sub["realized_resid"].mean()) if len(sub) else None,
        }
    b = out["Blue"]["mean_realized_resid"] or 0.0
    g = out["Gray"]["mean_realized_resid"] or 0.0
    out["blue_minus_gray"] = b - g
    out["signal_valid"] = b > g
    return out
