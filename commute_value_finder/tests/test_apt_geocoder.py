import json
import pandas as pd
from src.apt_geocoder import load_dong_coords, geocode_complexes


def test_load_dong_coords(tmp_path):
    cache = {"강남구_삼성동": {"lat": 37.51, "lon": 127.05}, "노원구_상계동": None}
    p = tmp_path / "geocode_cache.json"
    p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    coords = load_dong_coords(p)
    assert coords[("강남구", "삼성동")] == (37.51, 127.05)
    assert ("노원구", "상계동") not in coords  # null 제외


def test_geocode_complexes_uses_geocoder_and_dong_fallback(tmp_path):
    complexes = pd.DataFrame(
        {"구": ["강남구", "강남구"], "법정동": ["삼성동", "삼성동"],
         "아파트명": ["힐스테이트", "없는단지"]}
    )
    dong_coords = {("강남구", "삼성동"): (37.51, 127.05)}

    def fake_geocode(api_key, query):
        return (37.512, 127.056) if "힐스테이트" in query else None

    out = geocode_complexes(complexes, api_key="KEY", dong_coords=dong_coords,
                            cache_path=tmp_path / "apt.json",
                            geocoder=fake_geocode, sleeper=lambda s: None)
    assert out.loc[0, "lat"] == 37.512   # 단지 좌표
    assert out.loc[1, "lat"] == 37.51    # 동 좌표 폴백
    saved = json.loads((tmp_path / "apt.json").read_text("utf-8"))
    assert "강남구_삼성동_힐스테이트" in saved


def test_geocode_complexes_respects_max_calls(tmp_path):
    complexes = pd.DataFrame(
        {"구": ["강남구"] * 3, "법정동": ["삼성동"] * 3, "아파트명": ["A", "B", "C"]}
    )
    dong_coords = {("강남구", "삼성동"): (37.51, 127.05)}
    calls = {"n": 0}

    def fake_geocode(api_key, query):
        calls["n"] += 1
        return (37.5, 127.0)

    out = geocode_complexes(complexes, "KEY", dong_coords, tmp_path / "c.json",
                            geocoder=fake_geocode, max_calls=1, sleeper=lambda s: None)
    assert calls["n"] == 1               # API 1회만
    assert out["lat"].notna().all()      # 나머지는 동 좌표 폴백
