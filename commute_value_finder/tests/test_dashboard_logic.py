import pandas as pd
from src.dashboard_logic import (
    zone_label, risk_flags, naver_land_url, hogangnono_url, rank_candidates,
)


def test_zone_label():
    assert zone_label("Blue") == "저평가 후보 (확인 필요)"
    assert zone_label("Gray") == "적정"
    assert zone_label("Red") == "프리미엄"


def test_risk_flags():
    base = {"n": 12, "confidence": "높음", "deviation_pct": -20.0, "subway_dist_km": 0.3}
    assert risk_flags(base) == []
    assert any("극단" in f for f in risk_flags({**base, "deviation_pct": -70.0}))
    assert any("거래" in f for f in risk_flags({**base, "n": 3}))
    assert any("역" in f for f in risk_flags({**base, "subway_dist_km": 1.5}))


def test_linkout_urls():
    n = naver_land_url("노원구", "상계동", "A아파트")
    h = hogangnono_url("A아파트")
    assert n.startswith("https://") and "A아파트" in n
    assert h.startswith("https://") and "A아파트" in h


def test_rank_candidates_filters_and_sorts():
    zones = pd.DataFrame({
        "구": ["노원구", "도봉구", "강남구"],
        "법정동": ["상계동", "창동", "삼성동"],
        "아파트명": ["A", "B", "C"],
        "zone": ["Blue", "Blue", "Gray"],
        "commute_minutes": [40, 70, 20],
        "avg_price_per_sqm": [1500.0, 1400.0, 5000.0],
        "final_resid": [-0.2, -0.4, 0.0],
    })
    out = rank_candidates(zones, max_commute=60, max_price=3000, top_n=10)
    assert list(out["아파트명"]) == ["A"]
