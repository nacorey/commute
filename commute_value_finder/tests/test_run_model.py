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
