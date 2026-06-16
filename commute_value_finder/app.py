"""Commute-Value Finder — Streamlit 대시보드 (단지 단위 + 정직성 레이어)."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium
from folium.plugins import MarkerCluster
from dotenv import load_dotenv

from src.dashboard_logic import (
    zone_label, risk_flags, naver_land_url, hogangnono_url, rank_candidates,
)
from src.transit_projects import load_projects
from src.subway_access import load_stations
from src.briefing import build_context, get_briefing, rule_based_briefing

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Commute-Value Finder", page_icon="🏠", layout="wide")

ZONE_COLOR = {"Blue": "#2563EB", "Gray": "#94A3B8", "Red": "#EF4444"}


@st.cache_data
def load_zones():
    p = ROOT / "data" / "complex_zones.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


zones = load_zones()
if zones.empty:
    st.error("데이터가 없습니다. `py run_model.py`를 먼저 실행하세요.")
    st.stop()

st.title("🏠 Commute-Value Finder")
st.caption("통근·지하철 접근성 대비 가격이 낮은 단지를 '확인 필요 후보'로 스크리닝합니다.")
st.warning(
    "**이 도구는 예측기가 아니라 스크리너입니다.** 백테스트 결과, 저평가 후보가 "
    "향후 가격 상승을 예측하지는 못했습니다(통근·지하철 대비 '지속적으로 싼' 곳을 찾는 용도). "
    "또한 잔차에 공간 자기상관이 유의(Moran's I≈0.57)해, 학군 등 누락 요인의 영향이 섞여 "
    "있을 수 있습니다. 후보는 반드시 **직접 확인**하세요."
)

with st.sidebar:
    st.subheader("필터")
    max_commute = st.slider("통근 시간 상한 (분)", 15, 90, 50)
    max_price = st.slider("평당가 상한 (만원/㎡)", 500, 8000, 3000, step=100)
    show_blue = st.checkbox("Blue (저평가 후보)", True)
    show_gray = st.checkbox("Gray (적정)", False)
    show_red = st.checkbox("Red (프리미엄)", False)
    show_subway = st.toggle("지하철역 표시", False)
    show_transit = st.toggle("교통호재(GTX·신설) 표시", True)
    st.divider()
    st.subheader("AI 브리핑")
    budget = st.text_input("예산", "전세 5억")
    do_brief = st.button("브리핑 생성", use_container_width=True)

sel_zones = [z for z, on in [("Blue", show_blue), ("Gray", show_gray), ("Red", show_red)] if on]
filtered = zones[
    (zones["commute_minutes"] <= max_commute)
    & (zones["avg_price_per_sqm"] <= max_price)
    & (zones["zone"].isin(sel_zones or ["Blue"]))
].copy()
blue_f = zones[(zones["zone"] == "Blue") & (zones["commute_minutes"] <= max_commute)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("필터된 단지", f"{len(filtered):,}")
c2.metric("Blue 후보", f"{int((zones['zone']=='Blue').sum()):,}")
c3.metric("Blue 평균 평당가",
          f"{blue_f['avg_price_per_sqm'].mean():,.0f}" if len(blue_f) else "—")
c4.metric("Blue 평균 통근",
          f"{blue_f['commute_minutes'].mean():.0f}분" if len(blue_f) else "—")

tab_map, tab_rank, tab_brief, tab_model = st.tabs(
    ["지도", "후보 랭킹", "AI 브리핑", "모델·검증"])

with tab_map:
    m = folium.Map(location=[37.5665, 126.978], zoom_start=11, tiles="cartodbpositron")
    cap = filtered.head(1500)
    cluster = MarkerCluster().add_to(m)
    for _, r in cap.iterrows():
        color = ZONE_COLOR.get(r["zone"], "#888")
        flags = risk_flags(r)
        popup = (f"<b>{r['구']} {r['법정동']} {r['아파트명']}</b><br>"
                 f"{zone_label(r['zone'])}<br>"
                 f"통근 {int(r['commute_minutes'])}분 | 평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡<br>"
                 f"통근·지하철대비 {r['deviation_pct']:+.1f}% | 역 {r['subway_dist_km']:.2f}km<br>"
                 f"거래 {int(r['n'])}건 ({r['confidence']})"
                 + (f"<br><span style='color:#b45309'>⚠ {' / '.join(flags)}</span>" if flags else ""))
        folium.CircleMarker([r["lat"], r["lon"]], radius=5, color=color,
                            fill=True, fill_color=color, fill_opacity=0.6,
                            popup=folium.Popup(popup, max_width=300)).add_to(cluster)
    if show_subway:
        for s in load_stations(ROOT / "data" / "subway_stations.json"):
            folium.CircleMarker([s["lat"], s["lon"]], radius=2, color="#10B981",
                                fill=True, fill_opacity=0.7).add_to(m)
    if show_transit:
        for p in load_projects():
            folium.Marker([p["lat"], p["lon"]],
                          tooltip=f"{p['name']} ({p.get('open_year','-')} 예정)",
                          icon=folium.Icon(color="purple", icon="train", prefix="fa")).add_to(m)
    if len(filtered) > len(cap):
        st.caption(f"지도에는 {len(cap):,}개만 표시(성능). 전체 {len(filtered):,}개는 '후보 랭킹' 탭 참고.")
    components.html(m._repr_html_(), height=560)

with tab_rank:
    st.subheader("저평가 후보 랭킹 (확인 필요)")
    ranked = rank_candidates(zones, max_commute, max_price, top_n=30)
    if ranked.empty:
        st.info("조건에 맞는 Blue 후보가 없습니다. 필터를 완화하세요.")
    else:
        for _, r in ranked.iterrows():
            flags = risk_flags(r)
            with st.container(border=True):
                cc1, cc2 = st.columns([3, 1])
                with cc1:
                    st.markdown(f"**{r['구']} {r['법정동']} {r['아파트명']}**  "
                                f"`{r['confidence']}`")
                    st.caption(
                        f"통근 {int(r['commute_minutes'])}분 · "
                        f"평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡ · "
                        f"통근·지하철대비 {r['deviation_pct']:+.1f}% · "
                        f"역 {r['subway_dist_km']:.2f}km · 거래 {int(r['n'])}건")
                    if flags:
                        st.markdown("⚠ " + " / ".join(flags))
                with cc2:
                    st.link_button("네이버부동산", naver_land_url(r["구"], r["법정동"], r["아파트명"]),
                                   use_container_width=True)
                    st.link_button("호갱노노", hogangnono_url(r["아파트명"]),
                                   use_container_width=True)

with tab_brief:
    if do_brief:
        prefs = {"budget_type": "전세" if "전세" in budget else "매매",
                 "budget_amount": 50000, "max_commute": max_commute,
                 "priorities": ["통근 우선"]}
        ctx, cand = build_context(zones, prefs)
        with st.spinner("gpt-5.4-mini 브리핑 생성 중..."):
            api_key = os.getenv("OPENAI_API_KEY")
            result = get_briefing(ctx, prefs, api_key) if api_key else None
            if result is None:
                st.info("LLM 호출 실패/미설정 → 규칙기반 추천")
                result = rule_based_briefing(cand, prefs)
        st.session_state["brief"] = result
    if "brief" in st.session_state:
        b = st.session_state["brief"]
        for rec in b["recommendations"]:
            with st.container(border=True):
                st.markdown(f"**{rec['rank']}. {rec['구']} {rec['법정동']} {rec['아파트명']}**")
                rs = rec["reason"]
                st.write(f"💰 {rs['가격']}")
                st.write(f"🚇 {rs['통근']}")
                st.write(f"📊 {rs['거래신뢰도']}")
                st.write(f"⚠ {rs['주의']}")
                if rec.get("risk_notes"):
                    st.caption("리스크: " + ", ".join(rec["risk_notes"]))
        st.caption(b["disclaimer"])
    else:
        st.info("사이드바에서 **브리핑 생성**을 누르세요.")

with tab_model:
    st.subheader("모델 구조")
    st.markdown(
        "- **2단계 헤도닉:** 면적·평형·층·연식·시점을 통제해 가격 잔차를 구하고, "
        "단지별로 축소추정한 입지가치지수를 통근·지하철 접근성으로 회귀해 저평가를 판정.\n"
        "- **Blue 게이트:** 잔차 + 거래건수 + 최근성 + 신뢰도를 함께 만족해야 후보로 확정.")
    st.subheader("검증 (정직성)")
    st.markdown(
        "- **백테스트:** 저평가 후보가 향후 상승을 예측하지 **못함** → 스크리닝 용도.\n"
        "- **Moran's I ≈ 0.57 (p≈0.005):** 잔차의 공간 자기상관이 유의 → 누락변수(학군 등) 영향 가능.\n"
        "- **확인 필요:** 학군·향·소음·재건축·세대수는 데이터에 없으므로 직접 확인.")
