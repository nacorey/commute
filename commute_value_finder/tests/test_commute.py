import json
import pandas as pd
from src.commute import load_dong_commute, KakaoDrivingEstimator


def test_load_dong_commute(tmp_path):
    cache = {"강남구_삼성동": 12, "노원구_상계동": 48}
    p = tmp_path / "commute_cache.json"
    p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    df = load_dong_commute(p)
    assert set(df.columns) == {"구", "법정동", "commute_minutes"}
    assert len(df) == 2
    row = df.set_index(["구", "법정동"]).loc[("강남구", "삼성동")]
    assert int(row["commute_minutes"]) == 12


def test_load_dong_commute_missing_file(tmp_path):
    df = load_dong_commute(tmp_path / "nope.json")
    assert list(df.columns) == ["구", "법정동", "commute_minutes"]
    assert df.empty


def test_kakao_estimator_converts_seconds_to_minutes():
    calls = []
    def fake_fetch(api_key, o_lat, o_lng, d_lat, d_lng):
        calls.append((o_lat, o_lng))
        return 600  # seconds
    est = KakaoDrivingEstimator("KEY", 37.5, 127.0, fetch=fake_fetch, sleeper=lambda s: None)
    assert est.minutes(37.6, 127.1) == 10
    assert len(calls) == 1


def test_kakao_estimator_daily_limit():
    est = KakaoDrivingEstimator("KEY", 37.5, 127.0, fetch=lambda *a: 600,
                                sleeper=lambda s: None, daily_limit=1)
    assert est.minutes(1, 1) == 10
    assert est.minutes(1, 1) is None


def test_kakao_estimator_retries_then_gives_up():
    attempts = {"n": 0}
    def flaky(*a):
        attempts["n"] += 1
        raise RuntimeError("boom")
    est = KakaoDrivingEstimator("KEY", 37.5, 127.0, fetch=flaky, sleeper=lambda s: None)
    assert est.minutes(1, 1) is None
    assert attempts["n"] == 3
