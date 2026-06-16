import json
import pandas as pd
from src.briefing import build_context, get_briefing, rule_based_briefing


def _zones():
    return pd.DataFrame(
        {
            "구": ["노원구", "도봉구"],
            "법정동": ["상계동", "창동"],
            "아파트명": ["A아파트", "B아파트"],
            "zone": ["Blue", "Blue"],
            "commute_minutes": [45, 50],
            "avg_price_per_sqm": [1500.0, 1400.0],
            "final_resid": [-0.3, -0.25],
            "deviation_pct": [-30.0, -25.0],
            "n": [12, 8],
            "confidence": ["높음", "보통"],
            "subway_dist_km": [0.3, 0.6],
        }
    )


def _prefs():
    return {"budget_type": "전세", "budget_amount": 50000,
            "max_commute": 60, "priorities": ["통근 우선"]}


def test_build_context_filters_blue_and_lists_candidates():
    ctx, cand = build_context(_zones(), _prefs())
    assert "A아파트" in ctx
    assert len(cand) == 2
    assert (cand["zone"] == "Blue").all()


def test_get_briefing_parses_structured_json():
    payload = {
        "recommendations": [
            {"rank": 1, "구": "노원구", "법정동": "상계동", "아파트명": "A아파트",
             "reason": {"가격": "x", "통근": "y", "거래신뢰도": "z", "주의": "w"},
             "avg_price_per_sqm": 1500.0, "commute_min": 45, "residual": -0.3,
             "risk_notes": ["거래 적음"]}
        ],
        "disclaimer": "참고용",
    }
    fake_caller = lambda model, system, user: json.dumps(payload)
    out = get_briefing("ctx", _prefs(), "KEY", caller=fake_caller)
    assert out["recommendations"][0]["아파트명"] == "A아파트"
    assert out["disclaimer"] == "참고용"


def test_get_briefing_returns_none_on_error():
    def boom(model, system, user):
        raise RuntimeError("api down")
    out = get_briefing("ctx", _prefs(), "KEY", caller=boom)
    assert out is None


def test_rule_based_briefing_matches_schema_shape():
    _, cand = build_context(_zones(), _prefs())
    out = rule_based_briefing(cand, _prefs())
    assert "recommendations" in out and "disclaimer" in out
    assert len(out["recommendations"]) >= 1
    rec = out["recommendations"][0]
    assert set(["rank", "구", "법정동", "아파트명", "reason", "risk_notes"]).issubset(rec)
