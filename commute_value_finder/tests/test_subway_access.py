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
