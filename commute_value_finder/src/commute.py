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

import numpy as np
import pandas as pd
import requests
from src.subway_access import haversine_km

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


def calibrate_min_per_km(commute_df, dong_coords, ref_lat, ref_lng):
    """기존 통근(분)과 ref 위치까지의 직선거리로 분/km 보정계수(중앙값) 추정.

    너무 가까운 동(<0.5km)은 비율이 불안정해 제외. 표본 없으면 3.0 기본.
    """
    ratios = []
    for _, r in commute_df.iterrows():
        c = dong_coords.get((r["구"], r["법정동"]))
        if not c:
            continue
        km = haversine_km(c[0], c[1], ref_lat, ref_lng)
        if km > 0.5 and r["commute_minutes"] > 0:
            ratios.append(r["commute_minutes"] / km)
    return float(np.median(ratios)) if ratios else 3.0


def straight_line_commute(dong_coords, dest_lat, dest_lng, min_per_km):
    """각 동 직선거리 × 보정계수 → 추정 통근(분) DataFrame.

    동 좌표 dict {(구,법정동): (lat,lon)} 기준. 최소 1분.
    """
    rows = []
    for (gu, dong), (lat, lon) in dong_coords.items():
        km = haversine_km(lat, lon, dest_lat, dest_lng)
        rows.append({"구": gu, "법정동": dong,
                     "commute_minutes": max(1, round(km * min_per_km))})
    return pd.DataFrame(rows)


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
