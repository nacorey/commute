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
