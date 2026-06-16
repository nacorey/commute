import pandas as pd
from src.preprocessor import preprocess
from src.value_model import fit_quality_model, aggregate_to_complex, classify_zones
from config import SHRINKAGE_K, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX, RECENCY_MONTHS


def test_full_pipeline_produces_zones(sample_transactions):
    clean = preprocess(sample_transactions)
    _, scored, _ = fit_quality_model(clean)
    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    commute = pd.DataFrame(
        {
            "구": ["강남구", "노원구"],
            "법정동": ["삼성동", "상계동"],
            "commute_minutes": [10, 50],
        }
    )
    latest_ym = int(clean["거래년월"].max())
    out = classify_zones(
        comp, commute, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
        RECENCY_MONTHS, latest_ym,
    )
    assert set(out["zone"]).issubset({"Blue", "Gray", "Red"})
    assert {"입지가치지수", "final_resid", "zone", "confidence"}.issubset(out.columns)


def test_pipeline_with_subway_feature(sample_transactions):
    from src.subway_access import add_subway_access
    clean = preprocess(sample_transactions)
    _, scored, _ = fit_quality_model(clean)
    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    comp["lat"] = 37.5
    comp["lon"] = 127.0
    stations = [{"name": "역", "lat": 37.51, "lon": 127.0}]
    comp = add_subway_access(comp, stations)
    commute = pd.DataFrame({"구": ["강남구"], "법정동": ["삼성동"], "commute_minutes": [10]})
    out = classify_zones(comp, commute, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
                         RECENCY_MONTHS, int(clean["거래년월"].max()))
    assert "subway_dist_km" in out.columns
    assert set(out["zone"]).issubset({"Blue", "Gray", "Red"})
