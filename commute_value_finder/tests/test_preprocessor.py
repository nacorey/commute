import numpy as np
import pandas as pd
from src.preprocessor import (
    coerce_amount,
    add_derived_columns,
    remove_invalid,
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
