"""통근 시간 기반 잔차 분석 및 Zone 분류 모듈 (Phase 2)

Step 1: 카카오 모빌리티 API로 통근 시간 수집
Step 2: 통근 시간 → 평당가 선형 회귀
Step 3: 잔차 ±1σ 기준 Blue/Gray/Red Zone 분류
Step 4: Zone 지도 시각화
Step 5: 분석 요약 출력
"""

import sys
import json
import time
import os
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests as req
import pandas as pd
import numpy as np
import folium
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from config import COMPANY_LAT, COMPANY_LNG, COMPANY_NAME
from src.utils import get_project_root


KAKAO_NAVI_URL = "https://apis-navi.kakaomobility.com/v1/directions"

_navi_checked = False
_navi_ok = False


# ────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    """두 좌표 간 직선거리 (km)"""
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return {}


def _save_json(data: dict, path: Path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ────────────────────────────────────────────────────────
# Step 1: 통근 시간 수집
# ────────────────────────────────────────────────────────

def _fetch_commute_kakao(kakao_key: str, lat: float, lon: float) -> int | None:
    """카카오 모빌리티 길찾기 API로 운전 시간 조회 (분)"""
    global _navi_checked, _navi_ok
    if _navi_checked and not _navi_ok:
        return None

    try:
        resp = req.get(
            KAKAO_NAVI_URL,
            params={
                "origin": f"{lon},{lat}",
                "destination": f"{COMPANY_LNG},{COMPANY_LAT}",
                "priority": "RECOMMEND",
                "car_fuel": "GASOLINE",
            },
            headers={"Authorization": f"KakaoAK {kakao_key}"},
            timeout=10,
        )
        _navi_checked = True

        if resp.status_code == 403:
            _navi_ok = False
            return None

        resp.raise_for_status()
        _navi_ok = True

        routes = resp.json().get("routes", [])
        if routes and routes[0].get("result_code") == 0:
            return round(routes[0]["summary"]["duration"] / 60)
    except Exception:
        pass
    return None


def get_commute_times(dong_stats: pd.DataFrame, kakao_key: str | None) -> pd.DataFrame:
    """각 동 → 회사 통근 시간 수집 (캐시 사용)"""
    cache_path = get_project_root() / "data" / "commute_cache.json"
    cache = _load_json(cache_path)

    df = dong_stats.copy()
    uncached = [
        (r["구"], r["법정동"], r["lat"], r["lon"])
        for _, r in df.iterrows()
        if f"{r['구']}_{r['법정동']}" not in cache
    ]

    if uncached:
        api_count, fallback_count = 0, 0
        print(f"새로 조회: {len(uncached)}개 동 (캐시: {len(df) - len(uncached)}개)")

        for gu, dong, lat, lon in tqdm(uncached, desc="통근 시간", unit="동"):
            key = f"{gu}_{dong}"
            minutes = None

            if kakao_key:
                minutes = _fetch_commute_kakao(kakao_key, lat, lon)
                if minutes is not None:
                    api_count += 1
                time.sleep(0.1)

            if minutes is None:
                dist = _haversine_km(lat, lon, COMPANY_LAT, COMPANY_LNG)
                minutes = round(dist / 0.3)
                fallback_count += 1

            cache[key] = minutes

        _save_json(cache, cache_path)
        print(f"  API 성공: {api_count}건, 직선거리 추정: {fallback_count}건")
    else:
        print(f"전체 {len(df)}개 동 캐시 적중 (API 호출 0건)")

    df["commute_minutes"] = df.apply(
        lambda r: cache.get(f"{r['구']}_{r['법정동']}"), axis=1
    ).astype(int)

    return df


# ────────────────────────────────────────────────────────
# Step 2: 선형 회귀 분석
# ────────────────────────────────────────────────────────

def run_regression(df: pd.DataFrame) -> tuple[LinearRegression, float]:
    """commute_minutes → avg_price_per_sqm 선형 회귀"""
    X = df[["commute_minutes"]].values
    y = df["avg_price_per_sqm"].values

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    coef = model.coef_[0]

    print(f"  회귀 계수: {coef:,.1f} (통근 1분 증가 → 평당가 {coef:+,.1f}만원)")
    print(f"  절편: {model.intercept_:,.1f}만원/㎡")
    print(f"  R² = {r2:.4f}")

    # 산점도 + 회귀선
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["commute_minutes"], y, c="#7f8c8d", alpha=0.5, s=30, edgecolors="none")

    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax.plot(x_line, model.predict(x_line), color="#E74C3C", linewidth=2,
            label=f"회귀선 (R²={r2:.3f})")

    ax.set_xlabel("통근 시간 (분)", fontsize=12)
    ax.set_ylabel("평균 평당가 (만원/㎡)", fontsize=12)
    ax.set_title("통근 시간 vs 평당가 — 선형 회귀", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out = get_project_root() / "output" / "regression_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  산점도 저장: {out}")

    return model, r2


# ────────────────────────────────────────────────────────
# Step 3: 잔차 분석 / Zone 분류
# ────────────────────────────────────────────────────────

def classify_zones(df: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    """잔차 ±1σ 기준 Blue/Gray/Red Zone 분류"""
    df = df.copy()
    predicted = model.predict(df[["commute_minutes"]].values)
    df["predicted_price"] = predicted
    df["residual"] = df["avg_price_per_sqm"] - predicted

    sigma = df["residual"].std()
    df["zone"] = "Gray"
    df.loc[df["residual"] < -sigma, "zone"] = "Blue"
    df.loc[df["residual"] > sigma, "zone"] = "Red"

    # 예측가 대비 편차(%)
    df["deviation_pct"] = (df["residual"] / df["predicted_price"] * 100).round(1)

    zone_labels = {"Blue": "저평가", "Gray": "적정가", "Red": "프리미엄"}
    for z in ["Blue", "Gray", "Red"]:
        sub = df[df["zone"] == z]
        print(f"  {z:5s} ({zone_labels[z]}): {len(sub):>3d}개 동 | "
              f"평균 통근 {sub['commute_minutes'].mean():4.0f}분 | "
              f"평균 평당가 {sub['avg_price_per_sqm'].mean():>7,.0f}만원/㎡")

    return df


# ────────────────────────────────────────────────────────
# Step 4: Zone 지도 시각화
# ────────────────────────────────────────────────────────

def make_zone_map(df: pd.DataFrame) -> folium.Map:
    """Blue/Gray/Red Zone 지도 생성"""
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")

    STYLE = {
        "Blue": {"color": "#2196F3", "radius": 12, "weight": 3, "label": "저평가"},
        "Red":  {"color": "#F44336", "radius": 12, "weight": 3, "label": "프리미엄"},
        "Gray": {"color": "#9E9E9E", "radius": 8,  "weight": 1, "label": "적정가"},
    }

    # ── Zone 마커 ──
    zone_fg = folium.FeatureGroup(name="Zone 분류", show=True)

    for _, r in df.iterrows():
        s = STYLE[r["zone"]]
        check = " ✓" if r["zone"] == "Blue" else ""

        popup_html = (
            f"<b>{r['구']} {r['법정동']}</b><br>"
            f"Zone: <span style='color:{s['color']};font-weight:bold'>"
            f"{r['zone']}{check} {s['label']}</span><br>"
            f"통근: {int(r['commute_minutes'])}분<br>"
            f"평당가: {r['avg_price_per_sqm']:,.0f}만원/㎡<br>"
            f"주변 평균 대비: {r['deviation_pct']:+.1f}% (잔차 기반)<br>"
            f"거래 건수: {int(r['transaction_count']):,}건"
        )

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=s["radius"],
            color=s["color"],
            fill=True,
            fill_color=s["color"],
            fill_opacity=0.6,
            weight=s["weight"],
            tooltip=f"{r['구']} {r['법정동']} ({r['zone']})",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(zone_fg)

    zone_fg.add_to(m)

    # ── 회사 위치 ──
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

    # ── 통근 반경 ──
    radius_fg = folium.FeatureGroup(name="통근 반경 (직선거리)", show=True)
    for r_m, color, label in [
        (10_000, "#3498DB", "10km (~30분)"),
        (20_000, "#95A5A6", "20km (~60분)"),
    ]:
        folium.Circle(
            location=[COMPANY_LAT, COMPANY_LNG],
            radius=r_m, color=color, fill=False,
            weight=2, dash_array="10 6", tooltip=f"직선거리 {label}",
        ).add_to(radius_fg)
    radius_fg.add_to(m)

    # ── Zone 범례 (우상단) ──
    legend_html = """
    <div style="position:fixed; top:60px; right:10px; z-index:1000;
         background:white; padding:10px 14px; border-radius:6px;
         border:1px solid #ccc; font-size:12px; line-height:1.8;
         box-shadow:2px 2px 6px rgba(0,0,0,.15);">
        <b style="font-size:13px;">Zone 분류</b><br>
        <span style="color:#2196F3;">&#11044;</span>
          Blue Zone: 통근 대비 저평가 추천<br>
        <span style="color:#F44336;">&#11044;</span>
          Red Zone: 학군·인프라 프리미엄<br>
        <span style="color:#9E9E9E;">&#11044;</span>
          Gray Zone: 시장 적정가<br>
        <hr style="margin:4px 0;border:none;border-top:1px solid #eee;">
        <small style="color:#888;">잔차 &plusmn;1&sigma; 기준 분류</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ────────────────────────────────────────────────────────
# Step 5: 분석 요약
# ────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, r2: float):
    print(f"\n{'='*60}")
    print("분석 요약")
    print(f"{'='*60}")
    print(f"전체 분석 동 수: {len(df)}개")
    print(f"R² = {r2:.4f}", end=" → ")
    if r2 >= 0.5:
        print("모델 설명력 양호")
    elif r2 >= 0.3:
        print("모델 설명력 보통 (통근 외 요인도 상당)")
    else:
        print("모델 설명력 낮음 (통근 외 요인이 더 큰 영향)")

    blue = df[df["zone"] == "Blue"].sort_values("commute_minutes")
    print(f"\nBlue Zone 상위 10개 동 (통근 시간 짧은 순):")
    print(f"  {'동이름':<18s} {'통근':>6s} {'평당가':>12s} {'잔차':>10s}")
    print(f"  {'-'*50}")
    for _, r in blue.head(10).iterrows():
        name = f"{r['구']} {r['법정동']}"
        print(f"  {name:<18s} {int(r['commute_minutes']):>4d}분"
              f" {r['avg_price_per_sqm']:>10,.0f}"
              f" {r['residual']:>+10,.0f}")


# ────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────

def main():
    """Phase 2 메인: 통근 수집 → 회귀 → Zone 분류 → 지도"""
    from dotenv import load_dotenv
    load_dotenv(get_project_root() / ".env")

    data_dir = get_project_root() / "data"
    geo_path = data_dir / "seoul_apt_geocoded.csv"

    if not geo_path.exists():
        print("[ERROR] 데이터 없음. map_builder.py를 먼저 실행하세요.")
        return

    df = pd.read_csv(geo_path)
    print(f"원본 데이터: {len(df):,}건\n")

    # 동별 집계
    dong = df.groupby(["구", "법정동"]).agg(
        avg_price_per_sqm=("price_per_sqm", "mean"),
        median_price=("거래금액", "median"),
        transaction_count=("거래금액", "count"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()
    print(f"동별 집계: {len(dong)}개 동\n")

    # Step 1
    kakao_key = os.getenv("KAKAO_API_KEY")
    if kakao_key and kakao_key.startswith("여기에"):
        kakao_key = None

    print("=" * 50)
    print("Step 1: 통근 시간 수집")
    print("=" * 50)
    dong = get_commute_times(dong, kakao_key)

    commute_path = data_dir / "dong_with_commute.csv"
    dong.to_csv(commute_path, index=False, encoding="utf-8-sig")
    print(f"저장: {commute_path}")
    print(f"통근 시간 범위: {dong['commute_minutes'].min()} ~ "
          f"{dong['commute_minutes'].max()}분\n")

    # Step 2
    print("=" * 50)
    print("Step 2: 선형 회귀 분석")
    print("=" * 50)
    model, r2 = run_regression(dong)
    print()

    # Step 3
    print("=" * 50)
    print("Step 3: Zone 분류 (잔차 ±1σ)")
    print("=" * 50)
    dong = classify_zones(dong, model)

    # Zone 분류 결과 저장 (llm_briefing.py에서 로드)
    dong.to_csv(commute_path, index=False, encoding="utf-8-sig")
    print()

    # Step 4
    print("=" * 50)
    print("Step 4: Zone 지도 생성")
    print("=" * 50)
    zone_map = make_zone_map(dong)

    output_dir = get_project_root() / "output"
    output_dir.mkdir(exist_ok=True)
    map_path = output_dir / "map_phase2.html"
    zone_map.save(str(map_path))

    vc = dong["zone"].value_counts()
    print(f"지도 저장: {map_path}")
    print(f"  Blue: {vc.get('Blue', 0)}개 | "
          f"Gray: {vc.get('Gray', 0)}개 | "
          f"Red: {vc.get('Red', 0)}개")

    # Step 5
    print_summary(dong, r2)


if __name__ == "__main__":
    main()
