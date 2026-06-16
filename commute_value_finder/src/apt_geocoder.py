"""아파트 단지 단위 지오코딩 (카카오 키워드 검색, 캐시 + 동 좌표 폴백)."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def load_dong_coords(path):
    """geocode_cache.json('구_법정동' -> {lat,lon}) → {(구,동): (lat,lon)}."""
    path = Path(path)
    if not path.exists():
        return {}
    cache = json.loads(path.read_text("utf-8"))
    coords = {}
    for key, val in cache.items():
        if val and "lat" in val:
            gu, dong = key.split("_", 1)
            coords[(gu, dong)] = (val["lat"], val["lon"])
    return coords


def _kakao_keyword(api_key, query):
    resp = requests.get(KAKAO_KEYWORD_URL, params={"query": query, "size": 1},
                        headers={"Authorization": f"KakaoAK {api_key}"}, timeout=10)
    if resp.status_code == 403:
        raise PermissionError("Kakao 403")
    resp.raise_for_status()
    docs = resp.json().get("documents", [])
    if docs:
        return (float(docs[0]["y"]), float(docs[0]["x"]))
    return None


def geocode_complexes(complexes, api_key, dong_coords, cache_path,
                      geocoder=None, max_calls=None, sleeper=None):
    """각 단지 좌표 확보: 캐시 → 키워드 API → 동 좌표 폴백.

    df 순서대로 진행하며 max_calls까지만 API 호출(나머지는 폴백).
    반환: complexes + lat, lon 컬럼.
    """
    geocoder = geocoder or _kakao_keyword
    sleep = sleeper or time.sleep
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}

    calls = 0
    lats, lons = [], []
    for _, r in complexes.iterrows():
        gu, dong, apt = r["구"], r["법정동"], r["아파트명"]
        key = f"{gu}_{dong}_{apt}"
        coord = None
        if cache.get(key):
            coord = tuple(cache[key])
        elif api_key and (max_calls is None or calls < max_calls):
            try:
                coord = geocoder(api_key, f"{gu} {dong} {apt}")
                calls += 1
                if coord:
                    cache[key] = list(coord)
                sleep(0.05)
            except Exception:
                coord = None
        if not coord:
            coord = dong_coords.get((gu, dong))  # 동 좌표 폴백
        lats.append(coord[0] if coord else None)
        lons.append(coord[1] if coord else None)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    out = complexes.copy()
    out["lat"] = lats
    out["lon"] = lons
    return out
