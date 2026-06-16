"""Folium 기반 인터랙티브 지도 생성 모듈 (Phase 1)

Step 1: 법정동 지오코딩 (카카오 API → Nominatim 폴백)
Step 2: 동별 평균 평당가 집계
Step 3: 4개 레이어 지도 생성 (히트맵, 동별 마커, 회사 위치, 통근 반경)
Step 4: 레이어 컨트롤 및 HTML 저장
"""

import sys
import json
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests as req
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from tqdm import tqdm

from config import COMPANY_LAT, COMPANY_LNG, COMPANY_NAME
from src.utils import get_project_root


# ────────────────────────────────────────────────────────
# Step 1: 지오코딩
# ────────────────────────────────────────────────────────

KAKAO_GEOCODE_URL = "https://dapi.kakao.com/v2/local/search/address.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_kakao_checked = False
_kakao_ok = False
_last_nom_ts = 0.0


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return {}


def _save_cache(cache: dict, path: Path):
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")


def _geocode_kakao(kakao_key: str, gu: str, dong: str) -> tuple[float, float] | None:
    """카카오 주소 검색 API"""
    global _kakao_checked, _kakao_ok
    if _kakao_checked and not _kakao_ok:
        return None

    try:
        resp = req.get(
            KAKAO_GEOCODE_URL,
            params={"query": f"서울특별시 {gu} {dong}"},
            headers={"Authorization": f"KakaoAK {kakao_key}"},
            timeout=10,
        )
        _kakao_checked = True
        if resp.status_code == 403:
            _kakao_ok = False
            return None
        resp.raise_for_status()
        _kakao_ok = True
        docs = resp.json().get("documents", [])
        if docs:
            return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception:
        pass
    return None


def _geocode_nominatim(gu: str, dong: str) -> tuple[float, float] | None:
    """OpenStreetMap Nominatim (rate limit: 1 req/sec)"""
    global _last_nom_ts

    wait = 1.05 - (time.time() - _last_nom_ts)
    if wait > 0:
        time.sleep(wait)
    _last_nom_ts = time.time()

    try:
        resp = req.get(
            NOMINATIM_URL,
            params={
                "q": f"서울특별시 {gu} {dong}",
                "format": "json",
                "limit": 1,
                "countrycodes": "kr",
            },
            headers={"User-Agent": "CommuteValueFinder/1.0"},
            timeout=10,
        )
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def geocode_dong(kakao_key: str | None, gu: str, dong: str) -> tuple[float, float] | None:
    """법정동 좌표 조회: 카카오 → Nominatim 순서"""
    if kakao_key:
        result = _geocode_kakao(kakao_key, gu, dong)
        if result:
            return result
    return _geocode_nominatim(gu, dong)


def add_coordinates(df: pd.DataFrame, kakao_key: str | None) -> pd.DataFrame:
    """모든 법정동에 lat, lon 좌표 추가 (캐시 사용)"""
    cache_path = get_project_root() / "data" / "geocode_cache.json"
    cache = _load_cache(cache_path)

    unique_dongs = df[["구", "법정동"]].drop_duplicates()
    uncached = [
        (r["구"], r["법정동"])
        for _, r in unique_dongs.iterrows()
        if f"{r['구']}_{r['법정동']}" not in cache
    ]

    if uncached:
        geocoder = "카카오 API" if kakao_key else "Nominatim"
        print(f"지오코더: {geocoder} (실패 시 Nominatim 폴백)")
        print(f"새로 조회: {len(uncached)}개 동 (캐시 적중: {len(unique_dongs) - len(uncached)}개)")

        failed = []
        for gu, dong in tqdm(uncached, desc="좌표 변환", unit="동"):
            key = f"{gu}_{dong}"
            result = geocode_dong(kakao_key, gu, dong)
            if result:
                cache[key] = {"lat": result[0], "lon": result[1]}
            else:
                cache[key] = None
                failed.append(f"{gu} {dong}")

        _save_cache(cache, cache_path)
        if failed:
            print(f"실패: {len(failed)}개 - {failed[:5]}{'...' if len(failed) > 5 else ''}")
    else:
        print(f"전체 {len(unique_dongs)}개 동 캐시 적중 (API 호출 0건)")

    # 좌표 매핑 (merge - 대용량에도 빠름)
    coord_records = [
        {"구": k.split("_", 1)[0], "법정동": k.split("_", 1)[1],
         "lat": v["lat"], "lon": v["lon"]}
        for k, v in cache.items() if v is not None
    ]
    coord_df = pd.DataFrame(coord_records)

    before = len(df)
    df = df.merge(coord_df, on=["구", "법정동"], how="inner")

    success = df[["구", "법정동"]].drop_duplicates().shape[0]
    total = len(unique_dongs)
    print(f"좌표 매핑: {success}/{total}개 동 ({before:,} → {len(df):,}건)")

    return df


# ────────────────────────────────────────────────────────
# Step 2: 동별 집계
# ────────────────────────────────────────────────────────

def aggregate_by_dong(df: pd.DataFrame) -> pd.DataFrame:
    """법정동 기준 통계 집계"""
    return df.groupby(["구", "법정동"]).agg(
        avg_price_per_sqm=("price_per_sqm", "mean"),
        median_price=("거래금액", "median"),
        transaction_count=("거래금액", "count"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()


# ────────────────────────────────────────────────────────
# Step 3-4: 지도 생성
# ────────────────────────────────────────────────────────

def make_base_map(dong_stats: pd.DataFrame) -> folium.Map:
    """히트맵 + 동별 마커 + 회사 위치 + 통근 반경 지도 생성"""
    m = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    stats = dong_stats.copy()

    # ── 레이어 1: 히트맵 (평당가) ───────────────────────
    heat_fg = folium.FeatureGroup(name="히트맵 (평당가)", show=True)

    pmin = stats["avg_price_per_sqm"].min()
    pmax = stats["avg_price_per_sqm"].max()
    stats["_w"] = (stats["avg_price_per_sqm"] - pmin) / (pmax - pmin + 1e-9)

    HeatMap(
        stats[["lat", "lon", "_w"]].values.tolist(),
        radius=20,
        blur=15,
        max_zoom=13,
        gradient={0.4: "blue", 0.65: "lime", 0.8: "orange", 1.0: "red"},
    ).add_to(heat_fg)
    heat_fg.add_to(m)

    # ── 레이어 2: 동별 마커 ─────────────────────────────
    marker_fg = folium.FeatureGroup(name="동별 마커", show=True)

    p30 = stats["avg_price_per_sqm"].quantile(0.30)
    p70 = stats["avg_price_per_sqm"].quantile(0.70)
    cnt_min = stats["transaction_count"].min()
    cnt_range = max(stats["transaction_count"].max() - cnt_min, 1)

    for _, r in stats.iterrows():
        # 색상: 백분위수 기준
        if r["avg_price_per_sqm"] <= p30:
            color = "#3498DB"  # 파랑 (하위 30%)
        elif r["avg_price_per_sqm"] <= p70:
            color = "#2ECC71"  # 초록 (중간 40%)
        else:
            color = "#E74C3C"  # 빨강 (상위 30%)

        # 반지름: 거래 건수 비례 (5~15)
        radius = 5 + (r["transaction_count"] - cnt_min) / cnt_range * 10

        tip = (
            f"<b>{r['구']} {r['법정동']}</b><br>"
            f"평균 평당가: {r['avg_price_per_sqm']:,.0f}만원/㎡<br>"
            f"중위 거래가: {r['median_price']:,.0f}만원<br>"
            f"거래 건수: {int(r['transaction_count']):,}건"
        )

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            tooltip=folium.Tooltip(tip),
            popup=folium.Popup(tip, max_width=250),
        ).add_to(marker_fg)

    marker_fg.add_to(m)

    # ── 레이어 3: 회사 위치 ─────────────────────────────
    company_fg = folium.FeatureGroup(name="회사 위치", show=True)
    folium.Marker(
        location=[COMPANY_LAT, COMPANY_LNG],
        tooltip=f"기준: {COMPANY_NAME}",
        popup=folium.Popup(
            f"<b>기준 회사 위치</b><br>{COMPANY_NAME}<br>"
            f"<small>(config.py에서 변경 가능)</small>",
            max_width=200,
        ),
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(company_fg)
    company_fg.add_to(m)

    # ── 레이어 4: 통근 반경 (직선거리) ──────────────────
    radius_fg = folium.FeatureGroup(name="통근 반경 (직선거리)", show=True)

    folium.Circle(
        location=[COMPANY_LAT, COMPANY_LNG],
        radius=10_000,
        color="#3498DB",
        fill=False,
        weight=2,
        dash_array="10 6",
        tooltip="직선거리 10km (~30분 추정)",
    ).add_to(radius_fg)

    folium.Circle(
        location=[COMPANY_LAT, COMPANY_LNG],
        radius=20_000,
        color="#95A5A6",
        fill=False,
        weight=2,
        dash_array="10 6",
        tooltip="직선거리 20km (~60분 추정)",
    ).add_to(radius_fg)

    radius_fg.add_to(m)

    # ── 범례 (HTML 오버레이) ────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:10px 14px; border-radius:6px;
         border:1px solid #ccc; font-size:12px; line-height:1.6;
         box-shadow:2px 2px 6px rgba(0,0,0,.15);">
        <b style="font-size:13px;">범례</b><br>
        <span style="color:#3498DB;">&#9679;</span> 하위 30% (저가)<br>
        <span style="color:#2ECC71;">&#9679;</span> 중간 40%<br>
        <span style="color:#E74C3C;">&#9679;</span> 상위 30% (고가)<br>
        <hr style="margin:4px 0;border:none;border-top:1px solid #eee;">
        <span style="color:#3498DB;">- -</span> 10km (~30분)<br>
        <span style="color:#95A5A6;">- -</span> 20km (~60분)<br>
        <small style="color:#888;">직선거리 기준 추정<br>
        (실제 통근 시간은 Phase 2에서 계산)</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── 레이어 컨트롤 ──────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ────────────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────────────

def main():
    """Phase 1 메��: 지오코딩 → 집계 → 지도 생성"""
    from dotenv import load_dotenv
    load_dotenv(get_project_root() / ".env")

    data_dir = get_project_root() / "data"
    csv_path = data_dir / "seoul_apt_transactions.csv"

    if not csv_path.exists():
        print(f"[ERROR] 데이터 파일 없음: {csv_path}")
        print("먼저 data_collector.py를 실행하세요.")
        return

    df = pd.read_csv(csv_path)
    print(f"원본 데이터: {len(df):,}건\n")

    # Step 1: 좌표 붙이기
    kakao_key = os.getenv("KAKAO_API_KEY")
    if kakao_key and kakao_key.startswith("여기에"):
        kakao_key = None

    print("=" * 50)
    print("Step 1: 법정동 좌표 변환")
    print("=" * 50)
    df_geo = add_coordinates(df, kakao_key)

    geo_path = data_dir / "seoul_apt_geocoded.csv"
    df_geo.to_csv(geo_path, index=False, encoding="utf-8-sig")
    print(f"저장: {geo_path}\n")

    # Step 2: 동별 집계
    print("=" * 50)
    print("Step 2: 동별 집계")
    print("=" * 50)
    dong_stats = aggregate_by_dong(df_geo)
    print(f"집계 완료: {len(dong_stats)}개 동")
    print(f"  평당가 범위: {dong_stats['avg_price_per_sqm'].min():,.0f} ~ "
          f"{dong_stats['avg_price_per_sqm'].max():,.0f} 만원/㎡")
    print(f"  거래 건수 범위: {dong_stats['transaction_count'].min()} ~ "
          f"{dong_stats['transaction_count'].max()}건\n")

    # Step 3-4: 지도 생성 및 저장
    print("=" * 50)
    print("Step 3-4: 지도 생성")
    print("=" * 50)
    fmap = make_base_map(dong_stats)

    output_dir = get_project_root() / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "map_phase1.html"
    fmap.save(str(output_path))

    print(f"지도 저장: {output_path}")
    print(f"  레이어 수: 4개")
    print(f"    - 히트맵 (평당가): {len(dong_stats)}개 포인트")
    print(f"    - 동별 마커: {len(dong_stats)}개")
    print(f"    - 회사 위치: 1개 ({COMPANY_NAME})")
    print(f"    - 통근 반경 원: 2개 (10km, 20km)")


if __name__ == "__main__":
    main()
