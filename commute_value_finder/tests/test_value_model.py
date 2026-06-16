import numpy as np
import pandas as pd
from src.preprocessor import preprocess
from src.value_model import fit_quality_model


def test_fit_quality_model_returns_residuals(sample_transactions):
    clean = preprocess(sample_transactions)
    model, scored, r2 = fit_quality_model(clean)
    assert "resid" in scored.columns
    assert len(scored) == len(clean)
    assert abs(scored["resid"].mean()) < 1e-6
    assert 0.0 <= r2 <= 1.0


from src.value_model import aggregate_to_complex


def test_aggregate_to_complex_shrinks_small_samples():
    # 단지 X: 거래 1건, 잔차 +1.0 (동 평균 쪽으로 끌려가야 함)
    scored = pd.DataFrame(
        {
            "구": ["강남구", "강남구", "강남구"],
            "법정동": ["삼성동", "삼성동", "삼성동"],
            "아파트명": ["X", "Y", "Y"],
            "거래년월": [202601, 202602, 202603],
            "resid": [1.0, 0.0, 0.0],
        }
    )
    comp = aggregate_to_complex(scored, k=5)
    x_row = comp[comp["아파트명"] == "X"].iloc[0]
    assert x_row["n"] == 1
    assert x_row["입지가치지수"] < 1.0
    assert x_row["last_ym"] == 202601


from src.value_model import ym_subtract_months, classify_zones


def test_ym_subtract_months():
    assert ym_subtract_months(202603, 6) == 202509
    assert ym_subtract_months(202601, 1) == 202512


def test_classify_zones_applies_blue_gate():
    # 두 단지 모두 강한 음의 최종잔차(저평가)지만,
    # LOWN은 거래건수 부족 → 게이트 탈락 → Gray, 후보 플래그.
    # 상계동에 HIGH1~3(높은 입지가치지수)을 추가해 회귀선을 위로 고정,
    # 그 결과 GOOD/LOWN이 진성 음의 잔차 아웃라이어가 된다.
    comp = pd.DataFrame(
        {
            "구": ["노원구"] * 5 + ["강남구"],
            "법정동": ["상계동"] * 5 + ["삼성동"],
            "아파트명": ["HIGH1", "HIGH2", "HIGH3", "GOOD", "LOWN", "PREM"],
            "입지가치지수": [0.50, 0.40, 0.30, -0.30, -0.30, 0.50],
            "n": [10, 10, 10, 12, 2, 12],
            "last_ym": [202603, 202603, 202603, 202603, 202603, 202603],
        }
    )
    commute = pd.DataFrame(
        {
            "구": ["노원구", "강남구"],
            "법정동": ["상계동", "삼성동"],
            "commute_minutes": [50, 10],
        }
    )
    out = classify_zones(
        comp, commute, sigma_mult=1.0, min_tx=5,
        recency_months=6, latest_ym=202603,
    )
    z = out.set_index("아파트명")["zone"]
    assert z["GOOD"] == "Blue"          # 게이트 통과
    assert z["LOWN"] == "Gray"          # 거래건수 부족 → 강등
    assert out.set_index("아파트명").loc["LOWN", "blue_candidate_lowconf"]
    assert "confidence" in out.columns


def test_classify_zones_handles_unmatched_commute():
    # 통근 데이터가 어떤 단지 동과도 매칭되지 않아도 크래시 없이 분류돼야 함
    comp = pd.DataFrame(
        {
            "구": ["A", "A", "A"],
            "법정동": ["x", "x", "x"],
            "아파트명": ["P", "Q", "R"],
            "입지가치지수": [-0.5, 0.0, 0.5],
            "n": [10, 10, 10],
            "last_ym": [202603, 202603, 202603],
        }
    )
    commute = pd.DataFrame({"구": ["B"], "법정동": ["y"], "commute_minutes": [30]})
    out = classify_zones(comp, commute, sigma_mult=1.0, min_tx=5,
                         recency_months=6, latest_ym=202603)
    assert set(out["zone"]).issubset({"Blue", "Gray", "Red"})
    assert out["final_resid"].notna().all()


def test_classify_zones_uses_subway_feature_when_present():
    # 한 동(통근 동일)에서 subway_dist_km만 다르면, 예측이 단지마다 달라야 함
    comp = pd.DataFrame(
        {
            "구": ["A", "A", "A", "A"],
            "법정동": ["x", "x", "x", "x"],
            "아파트명": ["P", "Q", "R", "S"],
            "입지가치지수": [-0.4, -0.4, 0.4, 0.4],
            "n": [10, 10, 10, 10],
            "last_ym": [202603, 202603, 202603, 202603],
            "subway_dist_km": [0.2, 2.0, 0.2, 2.0],
        }
    )
    commute = pd.DataFrame({"구": ["A"], "법정동": ["x"], "commute_minutes": [30]})
    out = classify_zones(comp, commute, sigma_mult=1.0, min_tx=5,
                         recency_months=6, latest_ym=202603)
    assert "subway_dist_km" in out.columns
    assert out["pred_idx"].nunique() > 1  # subway가 회귀에 반영됨
