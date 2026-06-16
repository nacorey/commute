# Commute + Spatial Features + Resolution Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 통근 계산을 교체 가능한 인터페이스 뒤로 옮기고, 단지 단위 지오코딩과 지하철 접근성 피처를 더해, 가치 모델이 "통근 + 지하철 접근성" 둘 다로 저평가를 판정하게 한다.

**Architecture:** `commute.py`(통근 인터페이스+Phase A 운전 추정), `subway_access.py`(최근접 역 거리), `apt_geocoder.py`(단지 지오코딩, 캐시+동 좌표 폴백)를 신설하고, `value_model.classify_zones`가 `subway_dist_km`가 있으면 2단계 회귀에 함께 넣도록 확장한다. `run_model.py`가 이들을 엮어 단지 좌표·지하철거리가 포함된 `complex_zones.csv`를 만든다.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn, requests, pytest. (모두 설치됨. 새 무거운 의존성 없음.)

---

## Environment Notes (구현자 필독)
- Work from `commute_value_finder/`. Python launcher is **`py`** (NOT python/python3). Run tests with `py -m pytest ...`.
- git identity is configured. Create/Use branch `feat/commute-spatial-resolution` (Task 0).
- 보유 데이터: `data/geocode_cache.json` (동 좌표 323개, 형식 `"구_법정동" -> {"lat","lon"}`), `data/subway_stations.json` (441개, 형식 `[{"name","lat","lon","line"}, ...]`), `data/commute_cache.json` (`"구_법정동" -> 분`), `data/seoul_apt_transactions.csv`.
- Plan 1 산출물: `src/preprocessor.py`, `src/value_model.py`(fit_quality_model/aggregate_to_complex/classify_zones), `run_model.py`, `config.py` 파라미터.

---

## File Structure

| 파일 | 책임 |
|---|---|
| `src/commute.py` (신규) | `load_dong_commute`, `CommuteEstimator` 인터페이스, `KakaoDrivingEstimator`(재시도·일일가드) |
| `src/subway_access.py` (신규) | haversine, 최근접 역 거리, `add_subway_access` |
| `src/apt_geocoder.py` (신규) | 단지 키워드 지오코딩, 캐시 + 동 좌표 폴백, max_calls 가드 |
| `src/value_model.py` (수정) | `classify_zones`가 `subway_dist_km` 있으면 2단계 회귀에 포함 |
| `run_model.py` (수정) | 지오코딩 + 지하철 접근성 + 통근 인터페이스 통합 |
| `config.py` (수정) | `MAX_GEOCODE_CALLS` |
| `tests/` | 신규 단위 테스트 |

---

## Task 0: 브랜치 생성

- [ ] **Step 1: feature 브랜치 생성**

Run:
```bash
cd "C:/Users/ubumi/Desktop/2. 코딩연습/0. 2026/cursor/0401_01_commute"
git checkout main && git checkout -b feat/commute-spatial-resolution
git branch --show-current
```
Expected: `feat/commute-spatial-resolution`

---

## Task 1: `commute.py` — 통근 인터페이스 + Phase A 추정

**Files:**
- Create: `commute_value_finder/src/commute.py`
- Test: `commute_value_finder/tests/test_commute.py`

- [ ] **Step 1: Write failing test** — create `tests/test_commute.py`:
```python
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
```

- [ ] **Step 2: Run** `py -m pytest tests/test_commute.py -v` → FAIL (no module).

- [ ] **Step 3: Implement** — create `src/commute.py`:
```python
"""통근 시간 추정 (Phase A: 카카오 운전시간).

CommuteEstimator 인터페이스 뒤에 구현을 둬, Phase B(GTFS 대중교통)로
드롭인 교체할 수 있게 한다.
"""
import sys
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

KAKAO_NAVI_URL = "https://apis-navi.kakaomobility.com/v1/directions"


def load_dong_commute(cache_path) -> pd.DataFrame:
    """commute_cache.json('구_법정동' -> 분)을 DataFrame으로."""
    cache_path = Path(cache_path)
    cols = ["구", "법정동", "commute_minutes"]
    if not cache_path.exists():
        return pd.DataFrame(columns=cols)
    cache = json.loads(cache_path.read_text("utf-8"))
    rows = []
    for key, minutes in cache.items():
        gu, dong = key.split("_", 1)
        rows.append({"구": gu, "법정동": dong, "commute_minutes": minutes})
    return pd.DataFrame(rows, columns=cols)


class CommuteEstimator(ABC):
    """주거지 좌표 → 회사까지 통근 분. Phase B에서 교체될 인터페이스."""
    @abstractmethod
    def minutes(self, lat: float, lon: float):
        ...


def _kakao_fetch(api_key, o_lat, o_lng, d_lat, d_lng):
    """카카오 모빌리티 운전 소요 '초' 반환 (없으면 None, 403은 예외)."""
    resp = requests.get(
        KAKAO_NAVI_URL,
        params={"origin": f"{o_lng},{o_lat}", "destination": f"{d_lng},{d_lat}",
                "priority": "RECOMMEND"},
        headers={"Authorization": f"KakaoAK {api_key}"}, timeout=10,
    )
    if resp.status_code == 403:
        raise PermissionError("Kakao 403")
    resp.raise_for_status()
    routes = resp.json().get("routes", [])
    if routes and routes[0].get("result_code") == 0:
        return routes[0]["summary"]["duration"]
    return None


class KakaoDrivingEstimator(CommuteEstimator):
    """카카오 운전시간 추정. 재시도(backoff) + 일일 호출 가드."""
    def __init__(self, api_key, dest_lat, dest_lng, fetch=None, sleeper=None,
                 daily_limit=300000, max_retries=3, backoff_base=0.5):
        self.api_key = api_key
        self.dest_lat = dest_lat
        self.dest_lng = dest_lng
        self._fetch = fetch or _kakao_fetch
        self._sleep = sleeper or time.sleep
        self.daily_limit = daily_limit
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.calls = 0

    def minutes(self, lat, lon):
        if self.calls >= self.daily_limit:
            return None
        for attempt in range(self.max_retries):
            try:
                seconds = self._fetch(self.api_key, lat, lon, self.dest_lat, self.dest_lng)
                self.calls += 1
                if seconds is None:
                    return None
                return round(seconds / 60)
            except Exception:
                if attempt < self.max_retries - 1:
                    self._sleep(self.backoff_base * (2 ** attempt))
        return None
```

- [ ] **Step 4: Run** `py -m pytest tests/test_commute.py -v` → 5 PASS. Debug until green if needed.

- [ ] **Step 5: Commit**
```bash
git add src/commute.py tests/test_commute.py
git commit -m "feat: add commute estimator interface with kakao driving phase-A impl"
```

---

## Task 2: `subway_access.py` — 지하철 접근성 피처

**Files:**
- Create: `commute_value_finder/src/subway_access.py`
- Test: `commute_value_finder/tests/test_subway_access.py`

- [ ] **Step 1: Write failing test** — create `tests/test_subway_access.py`:
```python
import numpy as np
import pandas as pd
from src.subway_access import haversine_km, nearest_station_km, add_subway_access


def test_haversine_km_zero():
    assert haversine_km(37.5, 127.0, 37.5, 127.0) == 0.0


def test_haversine_km_about_one_km():
    d = haversine_km(37.5, 127.0, 37.509, 127.0)  # 위도 0.009도 ≈ 1km
    assert 0.9 < d < 1.1


def test_nearest_station_km():
    stations = [
        {"name": "가역", "lat": 37.50, "lon": 127.00},
        {"name": "나역", "lat": 37.60, "lon": 127.00},
    ]
    dist, name = nearest_station_km(37.51, 127.00, stations)
    assert name == "가역"
    assert dist < 2.0


def test_add_subway_access_orders_by_distance():
    stations = [{"name": "가역", "lat": 37.50, "lon": 127.00}]
    df = pd.DataFrame({"lat": [37.50, 37.60], "lon": [127.00, 127.00]})
    out = add_subway_access(df, stations)
    assert "subway_dist_km" in out.columns
    assert out.loc[0, "subway_dist_km"] < out.loc[1, "subway_dist_km"]


def test_add_subway_access_handles_missing_coords():
    stations = [{"name": "가역", "lat": 37.50, "lon": 127.00}]
    df = pd.DataFrame({"lat": [np.nan], "lon": [np.nan]})
    out = add_subway_access(df, stations)
    assert np.isnan(out.loc[0, "subway_dist_km"])
```

- [ ] **Step 2: Run** `py -m pytest tests/test_subway_access.py -v` → FAIL.

- [ ] **Step 3: Implement** — create `src/subway_access.py`:
```python
"""단지별 최근접 지하철역 도보거리(접근성) 피처."""
import sys
import json
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표 간 직선거리 (km)."""
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_stations(path):
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


def nearest_station_km(lat, lon, stations):
    """최근접 역까지 거리(km)와 역명."""
    best_d, best_name = float("inf"), None
    for s in stations:
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        if d < best_d:
            best_d, best_name = d, s.get("name")
    return best_d, best_name


def add_subway_access(df, stations):
    """df의 lat/lon에 대해 subway_dist_km, nearest_station 컬럼 추가.

    좌표 결측이거나 역 데이터가 없으면 NaN.
    """
    d = df.copy()

    def _calc(r):
        if pd.isna(r["lat"]) or pd.isna(r["lon"]) or not stations:
            return (float("nan"), None)
        return nearest_station_km(r["lat"], r["lon"], stations)

    res = d.apply(_calc, axis=1, result_type="expand")
    d["subway_dist_km"] = res[0]
    d["nearest_station"] = res[1]
    return d
```

- [ ] **Step 4: Run** `py -m pytest tests/test_subway_access.py -v` → 5 PASS. Debug if needed.

- [ ] **Step 5: Commit**
```bash
git add src/subway_access.py tests/test_subway_access.py
git commit -m "feat: add subway accessibility feature (nearest-station distance)"
```

---

## Task 3: `apt_geocoder.py` — 단지 지오코딩 (캐시 + 동 좌표 폴백)

**Files:**
- Create: `commute_value_finder/src/apt_geocoder.py`
- Test: `commute_value_finder/tests/test_apt_geocoder.py`

- [ ] **Step 1: Write failing test** — create `tests/test_apt_geocoder.py`:
```python
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
```

- [ ] **Step 2: Run** `py -m pytest tests/test_apt_geocoder.py -v` → FAIL.

- [ ] **Step 3: Implement** — create `src/apt_geocoder.py`:
```python
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
```

- [ ] **Step 4: Run** `py -m pytest tests/test_apt_geocoder.py -v` → 3 PASS. Debug if needed.

- [ ] **Step 5: Commit**
```bash
git add src/apt_geocoder.py tests/test_apt_geocoder.py
git commit -m "feat: add complex-level geocoding with cache and dong fallback"
```

---

## Task 4: `value_model.classify_zones` — 지하철 피처 추가 (하위호환)

**Files:**
- Modify: `commute_value_finder/src/value_model.py` (`classify_zones`의 2단계 회귀 부분만)
- Test: `commute_value_finder/tests/test_value_model.py` (테스트 추가)

- [ ] **Step 1: Write failing test** — append to END of `tests/test_value_model.py`:
```python
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
```

- [ ] **Step 2: Run** `py -m pytest tests/test_value_model.py::test_classify_zones_uses_subway_feature_when_present -v` → FAIL (pred_idx all equal because only commute used, and commute is constant → nunique == 1).

- [ ] **Step 3: Modify `classify_zones`** in `src/value_model.py`. Find this block:
```python
    model = LinearRegression().fit(d[["commute_minutes"]], d["입지가치지수"])
    d["pred_idx"] = model.predict(d[["commute_minutes"]])
```
Replace with:
```python
    feature_cols = ["commute_minutes"]
    if "subway_dist_km" in d.columns:
        subway_mean = d["subway_dist_km"].mean()
        if not pd.isna(subway_mean):
            d["subway_dist_km"] = d["subway_dist_km"].fillna(subway_mean)
            feature_cols.append("subway_dist_km")
    model = LinearRegression().fit(d[feature_cols], d["입지가치지수"])
    d["pred_idx"] = model.predict(d[feature_cols])
```

- [ ] **Step 4: Run** `py -m pytest tests/test_value_model.py -v` → ALL pass (6 total; the prior 5 still green since comp without subway_dist_km uses commute only). Debug if needed.

- [ ] **Step 5: Commit**
```bash
git add src/value_model.py tests/test_value_model.py
git commit -m "feat: include subway distance in stage-2 regression when available"
```

---

## Task 5: `run_model.py` 통합 + 실데이터 스모크

**Files:**
- Modify: `commute_value_finder/run_model.py` (전체 main 재구성)
- Modify: `commute_value_finder/config.py` (`MAX_GEOCODE_CALLS`)

- [ ] **Step 1: config에 파라미터 추가** — `commute_value_finder/config.py`의 "모델 정제 파라미터" 블록에 추가:
```python
MAX_GEOCODE_CALLS = 500               # 단지 지오코딩 1회 실행당 최대 API 호출(나머지 동 좌표 폴백)
```

- [ ] **Step 2: `run_model.py` 재작성** — `commute_value_finder/run_model.py` 전체를 아래로 교체:
```python
"""분석 엔진 엔드투엔드 실행 (Plan 1 + Plan 2).

거래 CSV → 정제 → 헤도닉 → 단지 집계 → 단지 지오코딩 → 지하철 접근성
→ 통근 결합 → 단지 단위 zone CSV.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

from config import (
    SHRINKAGE_K,
    ZONE_SIGMA,
    MIN_TRANSACTIONS_PER_COMPLEX,
    RECENCY_MONTHS,
    MAX_GEOCODE_CALLS,
)
from src.preprocessor import preprocess
from src.value_model import fit_quality_model, aggregate_to_complex, classify_zones
from src.commute import load_dong_commute
from src.subway_access import load_stations, add_subway_access
from src.apt_geocoder import load_dong_coords, geocode_complexes


def main():
    load_dotenv(ROOT / ".env")
    data_dir = ROOT / "data"
    src_csv = data_dir / "seoul_apt_transactions.csv"
    if not src_csv.exists():
        print(f"[ERROR] 데이터 없음: {src_csv}")
        return

    raw = pd.read_csv(src_csv)
    print(f"원본 거래: {len(raw):,}건")

    clean = preprocess(raw)
    print(f"정제 후: {len(clean):,}건")

    _, scored, r2 = fit_quality_model(clean)
    print(f"1단계 헤도닉 R² = {r2:.4f}")

    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    print(f"단지 수: {len(comp):,}개")

    # ── 단지 지오코딩 (거래 많은 단지 우선, 캐시 + 동 좌표 폴백) ──
    comp = comp.sort_values("n", ascending=False).reset_index(drop=True)
    dong_coords = load_dong_coords(data_dir / "geocode_cache.json")
    kakao_key = os.getenv("KAKAO_API_KEY")
    if kakao_key and kakao_key.startswith("여기에"):
        kakao_key = None
    comp = geocode_complexes(comp, kakao_key, dong_coords,
                             data_dir / "apt_geocode_cache.json",
                             max_calls=MAX_GEOCODE_CALLS)
    n_coords = int(comp["lat"].notna().sum())
    print(f"좌표 확보: {n_coords:,}/{len(comp):,} 단지")

    # ── 지하철 접근성 ──
    stations = load_stations(data_dir / "subway_stations.json")
    comp = add_subway_access(comp, stations)
    n_subway = int(comp["subway_dist_km"].notna().sum())
    if n_subway:
        print(f"지하철 접근성: {n_subway:,} 단지 | "
              f"평균 최근접역 {comp['subway_dist_km'].mean():.2f}km")

    # ── 통근 결합 + 분류 ──
    commute = load_dong_commute(data_dir / "commute_cache.json")
    latest_ym = int(clean["거래년월"].max())
    zones = classify_zones(comp, commute, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
                           RECENCY_MONTHS, latest_ym)

    out_path = data_dir / "complex_zones.csv"
    zones.to_csv(out_path, index=False, encoding="utf-8-sig")
    vc = zones["zone"].value_counts()
    print(f"저장: {out_path}")
    print(f"  Blue {vc.get('Blue', 0)} / Gray {vc.get('Gray', 0)} / Red {vc.get('Red', 0)}")
    print(f"  저평가 후보(저신뢰): {int(zones['blue_candidate_lowconf'].sum())}개")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 통합 테스트 갱신** — `tests/test_run_model.py`의 기존 테스트가 여전히 통과하는지 확인하고, 지하철 컬럼을 함께 검증하도록 보강. `tests/test_run_model.py` 끝에 추가:
```python
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
```

- [ ] **Step 4: Run** `py -m pytest tests/ -v` → ALL pass.

- [ ] **Step 5: REAL-DATA SMOKE** — run `py run_model.py`.
Expected: prints 원본/정제/R²/단지수, **좌표 확보 N/단지수**(geocode_cache 폴백으로 거의 전부 확보될 것), 지하철 접근성 평균 km, Blue/Gray/Red counts. Creates `data/complex_zones.csv` with `lat, lon, subway_dist_km, nearest_station` columns. Must complete WITHOUT errors. If Kakao key is invalid, geocoding falls back to dong coords (still works) — that's acceptable; report whether real complex coords were obtained or all fell back. Report FULL console output.

> NOTE: 단지 지오코딩은 최대 `MAX_GEOCODE_CALLS`(500)회만 실제 API를 호출하고 나머지는 동 좌표로 폴백한다(스모크 시간 제한). 전체 단지 정밀 지오코딩은 `MAX_GEOCODE_CALLS`를 올려 재실행하면 캐시로 이어서 완성된다.

- [ ] **Step 6: 산출물 확인**
Run:
```bash
py -c "import pandas as pd; d=pd.read_csv('data/complex_zones.csv'); print('cols:', [c for c in ['lat','lon','subway_dist_km','nearest_station','zone'] if c in d.columns]); print(d[['구','법정동','아파트명','subway_dist_km','commute_minutes','final_resid','zone']].sort_values('final_resid').head(8).to_string())"
```
Report output. Verify `subway_dist_km` is populated and `nearest_station` present.

- [ ] **Step 7: Commit**
```bash
git add run_model.py config.py tests/test_run_model.py
git commit -m "feat: integrate geocoding, subway access, and commute interface in pipeline"
```

---

## Self-Review (작성자 점검)

**스펙 커버리지 (Plan 2 = 구현순서 ③④⑤):**
- §5.1 commute 인터페이스 + Phase A + 캐시/백오프/일일가드 → Task 1 ✓
- §5.2 지하철 접근성(최근접 역 거리), 좌표계 haversine → Task 2 ✓
- §5.3 단지 지오코딩(키워드, 캐시, 동 폴백), max_calls 가드 → Task 3 ✓
- §5.4 2단계 회귀에 subway_dist_km 추가 → Task 4 ✓
- 통합 산출물(단지 좌표·지하철거리 포함 complex_zones.csv) → Task 5 ✓

**Plan 2 범위 밖(Plan 3):** 백테스트(validation.py), LLM(JSON 스키마), 교통호재, 대시보드 정직성 레이어, Moran's I 공간자기상관, 데이터 품질 로그(output/logs/), CLAUDE.md 정합성.

**타입/하위호환:** `classify_zones`는 `subway_dist_km` 컬럼이 있을 때만 회귀에 추가 → Plan 1 테스트(컬럼 없음) 그대로 통과. 통근 결합은 `classify_zones` 내부 merge 유지(시그니처 불변). 지오코딩은 항상 lat/lon 컬럼을 만들고(동 좌표 폴백), 결측 시 subway는 NaN→classify_zones가 mean으로 보정.

**플레이스홀더:** 없음 — 모든 코드 스텝에 실제 코드 포함.

---

## Execution Handoff
Plan 2 실행 후 Plan 3(검증·LLM·대시보드·호재)을 작성한다.
