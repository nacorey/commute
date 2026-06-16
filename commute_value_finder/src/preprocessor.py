"""실거래 데이터 정제·검증 레이어.

수집(data_collector)과 모델(value_model) 사이에서 정제를 전담한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import AREA_BANDS, PRICE_OUTLIER_PCT


def coerce_amount(value):
    """'84,500' 같은 문자열/숫자를 정수로. 실패 시 NaN."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, AttributeError):
        return np.nan


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """price_per_sqm, age, 거래년월(int), area_band 파생."""
    d = df.copy()
    d["거래금액"] = d["거래금액"].map(coerce_amount)
    d["전용면적"] = pd.to_numeric(d["전용면적"], errors="coerce")
    d["price_per_sqm"] = d["거래금액"] / d["전용면적"]
    d["age"] = pd.to_numeric(d["년"], errors="coerce") - pd.to_numeric(
        d["건축년도"], errors="coerce"
    )
    d["거래년월"] = (
        pd.to_numeric(d["년"], errors="coerce") * 100
        + pd.to_numeric(d["월"], errors="coerce")
    ).astype("Int64")

    bins = [-np.inf, *AREA_BANDS, np.inf]
    labels = ["~60", "60-85", "85-135", "135~"]
    d["area_band"] = pd.cut(d["전용면적"], bins=bins, labels=labels).astype(str)
    return d


def remove_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """비정상 면적(≤0)·연령(<0)·결측 평당가 제거."""
    d = df.copy()
    mask = (d["전용면적"] > 0) & (d["age"] >= 0) & d["price_per_sqm"].notna()
    return d[mask].reset_index(drop=True)


def remove_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """해제여부=='O'(계약 취소) 거래 제거. 컬럼 없으면 통과."""
    if "해제여부" not in df.columns:
        return df
    d = df.copy()
    keep = d["해제여부"].fillna("").astype(str).str.strip().str.upper() != "O"
    return d[keep].reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """동일 (아파트·법정동·층·면적·거래년월) 중복 제거."""
    keys = [c for c in ["아파트명", "법정동", "층", "전용면적", "거래년월"]
            if c in df.columns]
    return df.drop_duplicates(subset=keys).reset_index(drop=True)


def winsorize_price(df: pd.DataFrame, pct: tuple) -> pd.DataFrame:
    """price_per_sqm가 (low, high) 백분위 범위를 벗어나는 행을 제거(절단)한다.
    (값을 클램프하지 않고 이상치 행 자체를 떨어뜨린다.)
    """
    d = df.copy()
    lo, hi = np.percentile(d["price_per_sqm"].dropna(), pct)
    d = d[(d["price_per_sqm"] >= lo) & (d["price_per_sqm"] <= hi)]
    return d.reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """전체 정제 파이프라인."""
    d = add_derived_columns(df)
    d = remove_cancelled(d)
    d = remove_invalid(d)
    d = remove_duplicates(d)
    d = winsorize_price(d, PRICE_OUTLIER_PCT)
    return d
