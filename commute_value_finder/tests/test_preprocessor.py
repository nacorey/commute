import numpy as np
import pandas as pd
from src.preprocessor import (
    coerce_amount,
    add_derived_columns,
    remove_invalid,
    remove_cancelled,
    remove_duplicates,
    winsorize_price,
    preprocess,
)


def test_coerce_amount_handles_comma_string():
    assert coerce_amount("84,500") == 84500
    assert coerce_amount(84500) == 84500
    assert np.isnan(coerce_amount("abc"))


def test_add_derived_columns(sample_transactions):
    out = add_derived_columns(sample_transactions)
    assert out.loc[0, "price_per_sqm"] == 200000 / 84.0
    assert out.loc[0, "age"] == 2026 - 2010
    assert out.loc[0, "거래년월"] == 202603
    assert out.loc[0, "area_band"] == "60-85"
    assert out.loc[1, "area_band"] == "~60"


def test_remove_invalid_drops_nonpositive_area_and_negative_age():
    df = pd.DataFrame(
        {
            "전용면적": [84.0, 0.0, 84.0],
            "price_per_sqm": [100.0, 100.0, 100.0],
            "age": [5, 5, -3],
        }
    )
    out = remove_invalid(df)
    assert len(out) == 1


def test_remove_duplicates():
    df = pd.DataFrame(
        {
            "아파트명": ["A", "A"],
            "법정동": ["삼성동", "삼성동"],
            "층": [10, 10],
            "전용면적": [84.0, 84.0],
            "거래년월": [202603, 202603],
            "거래금액": [200000, 200000],
        }
    )
    assert len(remove_duplicates(df)) == 1


def test_remove_cancelled_drops_O_and_passes_when_column_absent():
    # 해제여부 == 'O'/'o' (계약 취소) 행 제거
    df = pd.DataFrame({"거래금액": [100, 200, 300], "해제여부": ["O", "", "o"]})
    out = remove_cancelled(df)
    assert len(out) == 1  # 빈 문자열 행만 유지
    # 컬럼이 없으면 그대로 통과
    df2 = pd.DataFrame({"거래금액": [1, 2]})
    assert len(remove_cancelled(df2)) == 2


def test_winsorize_price_clips_extremes():
    df = pd.DataFrame({"price_per_sqm": list(range(1, 101))})
    out = winsorize_price(df, (1, 99))
    assert out["price_per_sqm"].min() >= np.percentile(range(1, 101), 1)
    assert out["price_per_sqm"].max() <= np.percentile(range(1, 101), 99)


def test_preprocess_end_to_end(sample_transactions):
    out = preprocess(sample_transactions)
    assert "price_per_sqm" in out.columns
    assert "age" in out.columns
    assert "area_band" in out.columns
    assert (out["전용면적"] > 0).all()


def test_floor_coerced_and_invalid_floor_dropped():
    df = pd.DataFrame(
        {
            "구": ["A", "A"],
            "법정동": ["x", "x"],
            "아파트명": ["P", "Q"],
            "전용면적": [84.0, 84.0],
            "거래금액": [100000, 100000],
            "층": ["10", "B1"],          # B1 = 강제변환 불가 → 제거 대상
            "건축년도": [2000, 2000],
            "년": [2025, 2025],
            "월": [1, 2],
        }
    )
    out = preprocess(df)
    assert len(out) == 1                 # 비정상 층(B1) 행 제거
    assert float(out.iloc[0]["층"]) == 10.0
