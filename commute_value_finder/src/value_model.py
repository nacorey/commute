"""2단계 헤도닉 가치 모델.

1단계: 거래 건별 단위특성·평형·시점을 통제해 가격 잔차를 얻는다.
2단계: 단지 입지가치지수를 통근시간으로 회귀해 "저평가" 잔차를 얻는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """헤도닉 설계행렬: log(면적)+층+age+age²+평형더미+거래년월더미."""
    X = pd.concat(
        [
            np.log(df["전용면적"]).rename("log_area"),
            df["층"].astype(float).rename("floor"),
            df["age"].astype(float).rename("age"),
            (df["age"].astype(float) ** 2).rename("age2"),
            pd.get_dummies(df["area_band"], prefix="band", drop_first=True),
            pd.get_dummies(df["거래년월"].astype(str), prefix="ym", drop_first=True),
        ],
        axis=1,
    )
    return X.astype(float)


def fit_quality_model(df: pd.DataFrame):
    """1단계 헤도닉. 반환: (model, df+resid, r2).

    위치(구/동/단지) 변수는 넣지 않는다 — 위치 편차가 우리가 찾는 신호이므로.
    """
    d = df.copy()
    y = np.log(d["price_per_sqm"])
    X = _design_matrix(d)
    model = LinearRegression().fit(X, y)
    d["resid"] = y - model.predict(X)
    r2 = float(model.score(X, y))
    return model, d, r2
