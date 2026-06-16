"""Commute-Value Finder — Streamlit 대시보드 (Phase 4)"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests as req
import folium
from datetime import date
from dotenv import load_dotenv
import os

from config import (COMPANY_LAT, COMPANY_LNG, COMPANY_NAME,
                     SEOUL_CENTER, COMMUTE_RADIUS_M)

load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Commute-Value Finder",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────
# Theme: Dark Financial Terminal + Pretendard
# st.html()로 CSS 주입 (st.markdown의 <style> 스트리핑 방지)
# ────────────────────────────────────────────────────────

st.html("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css');

/* ═══ GLOBAL FONT ═══ */
*, html, body, .stApp, [class*="css"],
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-baseweb="tab"], [data-baseweb="select"],
button, input, select, textarea,
h1, h2, h3, h4, h5, h6 {
    font-family: "Pretendard Variable", Pretendard, -apple-system,
                 BlinkMacSystemFont, system-ui, sans-serif !important;
}

/* ═══ MAIN APP ═══ */
.stApp { background: #0B0F19 !important; }
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
}
footer { display: none !important; }

/* ═══ SIDEBAR ═══ */
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1F2A3D !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem !important;
}

/* ═══ TABS ═══ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1F2A3D !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #64748B !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 10px 22px !important;
    margin: 0 !important;
    transition: color 0.2s !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #94A3B8 !important;
}
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ═══ BUTTONS ═══ */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60A5FA, #3B82F6) !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.35) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: #111827 !important;
    color: #94A3B8 !important;
    border: 1px solid #1F2A3D !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #3B82F6 !important;
    color: #3B82F6 !important;
}

/* ═══ INPUTS ═══ */
[data-testid="stTextInput"] input {
    background: #1C2333 !important;
    border: 1px solid #1F2A3D !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-size: 0.82rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px rgba(59,130,246,0.3) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ═══ SLIDER ═══ */
[data-testid="stSlider"] [role="slider"] {
    background: #3B82F6 !important;
    border-color: #3B82F6 !important;
}

/* ═══ DIVIDERS ═══ */
hr { border-color: #1F2A3D !important; opacity: 0.6 !important; }

/* ═══ ALERTS ═══ */
[data-testid="stAlert"] {
    background: #1C2333 !important;
    border: 1px solid #1F2A3D !important;
    border-radius: 8px !important;
}

/* ═══ HEADINGS ═══ */
h2, h3, .stSubheader {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #64748B !important;
    font-size: 0.72rem !important;
}

/* ═══ DATAFRAME ═══ */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #1F2A3D !important;
}

/* ═══ MAP IFRAME ═══ */
.element-container iframe {
    border-radius: 10px !important;
    border: 1px solid #1F2A3D !important;
}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0B0F19; }
::-webkit-scrollbar-thumb { background: #2D3B54; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #64748B; }
</style>
""")


# ────────────────────────────────────────────────────────
# Helper: Metric Card (인라인 스타일 — CSS 의존 없음)
# ────────────────────────────────────────────────────────

def metric_card(label, value, unit="", sub="", accent="#3B82F6"):
    unit_html = (
        f'<span style="font-size:0.78rem;font-weight:400;'
        f'color:#94A3B8;margin-left:2px;">{unit}</span>'
    ) if unit else ""
    sub_html = (
        f'<div style="font-size:0.68rem;color:#94A3B8;margin-top:3px;">{sub}</div>'
    ) if sub else ""
    st.markdown(
        f'<div style="background:#111827;border:1px solid #1F2A3D;border-radius:10px;'
        f'padding:16px 18px;position:relative;overflow:hidden;border-top:2px solid {accent};">'
        f'<div style="font-size:0.68rem;font-weight:500;color:#64748B;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;'
        f'font-family:Pretendard Variable,Pretendard,sans-serif;">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:700;color:#F1F5F9;line-height:1.2;'
        f'letter-spacing:-0.02em;font-family:Pretendard Variable,Pretendard,sans-serif;">'
        f'{value}{unit_html}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────
# 데이터 로딩 (캐시)
# ────────────────────────────────────────────────────────

@st.cache_data
def load_dong_data():
    path = ROOT / "data" / "dong_with_commute.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "zone" not in df.columns:
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(df[["commute_minutes"]], df["avg_price_per_sqm"])
        pred = m.predict(df[["commute_minutes"]].values)
        df["predicted_price"] = pred
        df["residual"] = df["avg_price_per_sqm"] - pred
        s = df["residual"].std()
        df["zone"] = "Gray"
        df.loc[df["residual"] < -s, "zone"] = "Blue"
        df.loc[df["residual"] > s, "zone"] = "Red"
        df["deviation_pct"] = (df["residual"] / df["predicted_price"] * 100).round(1)
    return df


@st.cache_data
def load_raw_data():
    path = ROOT / "data" / "seoul_apt_geocoded.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_subway_stations():
    import json
    path = ROOT / "data" / "subway_stations.json"
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


@st.cache_data
def build_apt_lookup(raw_df):
    lookup = {}
    for (gu, dong), grp in raw_df.groupby(["구", "법정동"]):
        top = grp.nlargest(5, "거래금액")[["아파트명", "건축년도", "전용면적", "거래금액"]]
        lookup[f"{gu}_{dong}"] = top.values.tolist()
    return lookup


@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    kakao_key = os.getenv("KAKAO_API_KEY")
    if not kakao_key:
        return None
    for ep in [
        "https://dapi.kakao.com/v2/local/search/address.json",
        "https://dapi.kakao.com/v2/local/search/keyword.json",
    ]:
        try:
            r = req.get(ep, params={"query": address, "size": 1},
                        headers={"Authorization": f"KakaoAK {kakao_key}"}, timeout=5)
            docs = r.json().get("documents", [])
            if docs:
                return (float(docs[0]["y"]), float(docs[0]["x"]))
        except Exception:
            pass
    return None


# ────────────────────────────────────────────────────────
# 지하철 노선 색상
# ────────────────────────────────────────────────────────

LINE_COLORS = {
    "1": "#0052A4", "2": "#00A84D", "3": "#EF7C1C", "4": "#00A5DE",
    "5": "#996CAC", "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C",
    "9": "#BDB092",
}
DEFAULT_LINE_COLOR = "#888888"

# ────────────────────────────────────────────────────────
# 줌 보존 JavaScript
# ────────────────────────────────────────────────────────

MAP_STATE_JS = """
<script>
(function(){
  var KEY='cvf_map_state';
  var t=setInterval(function(){
    if(typeof L==='undefined')return;
    for(var k in window){try{
      if(window[k] instanceof L.Map){
        var m=window[k];clearInterval(t);
        try{
          var s=JSON.parse(localStorage.getItem(KEY));
          if(s&&s.lat)m.setView([s.lat,s.lng],s.zoom,{animate:false});
        }catch(e){}
        m.on('moveend',function(){
          var c=m.getCenter();
          localStorage.setItem(KEY,JSON.stringify({lat:c.lat,lng:c.lng,zoom:m.getZoom()}));
        });
        return;
      }
    }catch(e){}}
  },50);
  setTimeout(function(){clearInterval(t);},10000);
})();
</script>
"""

# ────────────────────────────────────────────────────────
# 지도 스타일 (Dark Map)
# ────────────────────────────────────────────────────────

ZONE_STYLE = {
    "Blue": {"color": "#60A5FA", "radius": 10, "weight": 2, "label": "저평가"},
    "Red":  {"color": "#F87171", "radius": 10, "weight": 2, "label": "프리미엄"},
    "Gray": {"color": "#94A3B8", "radius": 6,  "weight": 1, "label": "적정가"},
}

FOLIUM_DARK_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css');
.leaflet-popup-content-wrapper {
    background: #111827 !important;
    color: #F1F5F9 !important;
    border-radius: 10px !important;
    border: 1px solid #1F2A3D !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45) !important;
    font-family: "Pretendard Variable", Pretendard, sans-serif !important;
}
.leaflet-popup-tip { background: #111827 !important; }
.leaflet-popup-content { font-size: 12.5px !important; line-height: 1.65 !important; }
.leaflet-tooltip {
    background: #1C2333 !important;
    color: #F1F5F9 !important;
    border: 1px solid #1F2A3D !important;
    border-radius: 6px !important;
    font-family: "Pretendard Variable", Pretendard, sans-serif !important;
    font-size: 12px !important;
    padding: 5px 10px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35) !important;
}
.leaflet-tooltip-top::before { border-top-color: #1C2333 !important; }
.leaflet-tooltip-bottom::before { border-bottom-color: #1C2333 !important; }
.leaflet-tooltip-left::before { border-left-color: #1C2333 !important; }
.leaflet-tooltip-right::before { border-right-color: #1C2333 !important; }
</style>
"""


def build_map(dong, apt_lookup, show_subway, co_lat, co_lng, co_name):
    m = folium.Map(
        location=list(SEOUL_CENTER), zoom_start=11,
        tiles="cartodbdark_matter",
    )
    m.get_root().html.add_child(folium.Element(FOLIUM_DARK_CSS))

    for _, r in dong.iterrows():
        s = ZONE_STYLE.get(r["zone"], ZONE_STYLE["Gray"])

        key = f"{r['구']}_{r['법정동']}"
        apts = apt_lookup.get(key, [])
        apt_html = ""
        if apts:
            rows = ""
            for a in apts:
                name = str(a[0])[:14]
                year = int(a[1]) if pd.notna(a[1]) else "-"
                area = f"{a[2]:.0f}" if pd.notna(a[2]) else "-"
                price = f"{a[3]:,.0f}" if pd.notna(a[3]) else "-"
                rows += (
                    f'<tr style="border-bottom:1px solid #1F2A3D;">'
                    f'<td style="padding:3px 4px;">{name}</td>'
                    f'<td style="padding:3px 4px;text-align:center;">{year}</td>'
                    f'<td style="padding:3px 4px;text-align:right;">{area}</td>'
                    f'<td style="padding:3px 4px;text-align:right;">{price}</td></tr>'
                )
            apt_html = (
                '<div style="margin-top:8px;padding-top:8px;'
                'border-top:1px solid #1F2A3D;">'
                '<table style="font-size:11px;width:100%;border-collapse:collapse;'
                'color:#CBD5E1;">'
                '<tr style="border-bottom:1px solid #2D3B54;color:#94A3B8;">'
                '<td style="padding:3px 4px;font-weight:600;">아파트</td>'
                '<td style="padding:3px 4px;font-weight:600;text-align:center;">건축</td>'
                '<td style="padding:3px 4px;font-weight:600;text-align:right;">㎡</td>'
                '<td style="padding:3px 4px;font-weight:600;text-align:right;">만원</td></tr>'
                f'{rows}</table></div>'
            )

        dev = r.get("deviation_pct", 0)
        if dev < 0:
            interpret = (
                f"통근 {int(r['commute_minutes'])}분 기준, "
                f"평당가가 주변 대비 <b style='color:#60A5FA'>{abs(dev):.1f}% 저렴</b>"
            )
        elif dev > 0:
            interpret = (
                f"통근 {int(r['commute_minutes'])}분 기준, "
                f"평당가가 주변 대비 <b style='color:#F87171'>{dev:.1f}% 고가</b>"
            )
        else:
            interpret = "시장 적정가 수준"

        zone_bg = (
            "rgba(96,165,250,0.15)" if r["zone"] == "Blue"
            else "rgba(248,113,113,0.15)" if r["zone"] == "Red"
            else "rgba(148,163,184,0.15)"
        )
        popup = (
            f"<div style='min-width:230px;color:#E2E8F0;'>"
            f"<div style='font-weight:700;font-size:13px;margin-bottom:4px;'>"
            f"{r['구']} {r['법정동']}</div>"
            f"<span style='display:inline-block;padding:2px 8px;"
            f"border-radius:10px;font-size:10px;font-weight:600;"
            f"background:{zone_bg};"
            f"color:{s['color']};border:1px solid {s['color']}33;'>"
            f"{r['zone']} &middot; {s['label']}</span>"
            f"<div style='margin-top:6px;font-size:11.5px;color:#CBD5E1;line-height:1.6;'>"
            f"통근 <b>{int(r['commute_minutes'])}분</b>&ensp;|&ensp;"
            f"평당가 <b>{r['avg_price_per_sqm']:,.0f}</b> 만원/㎡<br>"
            f"<span style='font-size:10.5px;color:#94A3B8;'>{interpret}</span><br>"
            f"거래 {int(r['transaction_count']):,}건</div>"
            f"{apt_html}</div>"
        )

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=s["radius"], color=s["color"],
            fill=True, fill_color=s["color"],
            fill_opacity=0.55, weight=s["weight"],
            tooltip=f"{r['구']} {r['법정동']}  ({r['zone']})",
            popup=folium.Popup(popup, max_width=320),
        ).add_to(m)

    folium.Marker(
        location=[co_lat, co_lng], tooltip=f"기준: {co_name}",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(m)

    for radius_m, (color, label) in zip(
        COMMUTE_RADIUS_M,
        [("#3B82F6", "~30분"), ("#64748B", "~60분")]
    ):
        km = radius_m // 1000
        folium.Circle(
            location=[co_lat, co_lng], radius=radius_m, color=color,
            fill=False, weight=1.5, dash_array="8 6",
            tooltip=f"직선거리 {km}km ({label})",
        ).add_to(m)

    if show_subway:
        for s in load_subway_stations():
            line = s.get("line", "")
            color = LINE_COLORS.get(line, DEFAULT_LINE_COLOR)
            label = f"{line}호선" if line.isdigit() else (line + "선" if line else "")
            folium.CircleMarker(
                location=[s["lat"], s["lon"]], radius=3.5,
                color=color, fill=True, fill_color=color,
                fill_opacity=0.85, weight=1,
                tooltip=f"{s['name']} ({label})" if label else s["name"],
            ).add_to(m)

    from config import get_analysis_months
    months = get_analysis_months()
    date_label = f"{months[0][:4]}.{months[0][4:]}~{months[-1][:4]}.{months[-1][4:]}"
    legend = f"""
    <div style="position:fixed;bottom:24px;right:12px;z-index:1000;
         background:#111827ee;padding:12px 16px;border-radius:8px;
         border:1px solid #1F2A3D;font-size:11px;line-height:1.8;
         box-shadow:0 4px 20px rgba(0,0,0,.35);color:#CBD5E1;
         font-family:'Pretendard Variable',Pretendard,sans-serif;
         backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);">
        <div style="font-weight:700;font-size:12px;color:#F1F5F9;margin-bottom:4px;">
            Commute-Value Finder</div>
        <span style="color:#60A5FA;">●</span> Blue (저평가)&ensp;
        <span style="color:#94A3B8;">●</span> Gray (적정)&ensp;
        <span style="color:#F87171;">●</span> Red (프리미엄)<br>
        <span style="color:#EF4444;">★</span> 회사 위치&ensp;
        <span style="color:#64748B;font-size:10px;">{date_label}</span><br>
        <span style="color:#64748B;font-size:10px;">마커 클릭 → 아파트 상세</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.get_root().html.add_child(folium.Element(MAP_STATE_JS))

    return m


# ────────────────────────────────────────────────────────
# LLM 브리핑
# ────────────────────────────────────────────────────────

def run_briefing(dong, budget_text, max_commute, priorities_text, company_name):
    from src.llm_briefing import build_context, get_llm_briefing, _fallback_briefing, _parse_budget
    budget_type, budget_amount = _parse_budget(budget_text)
    prefs = {
        "budget_type": budget_type, "budget_amount": budget_amount,
        "max_commute": max_commute,
        "priorities": [p.strip() for p in priorities_text.split(",") if p.strip()]
                      or ["특별한 선호 없음"],
    }
    context, candidates = build_context(dong, prefs, company_name=company_name)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        result = get_llm_briefing(context, prefs, api_key)
        if result:
            return result
    return _fallback_briefing(candidates, prefs)


# ────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:11px;'
        'padding-bottom:1rem;margin-bottom:0.8rem;border-bottom:1px solid #1F2A3D;">'
        '<div style="display:flex;align-items:center;justify-content:center;'
        'width:34px;height:34px;border-radius:8px;'
        'background:linear-gradient(135deg,#3B82F6,#1D4ED8);'
        'color:#fff;font-weight:800;font-size:0.7rem;letter-spacing:0.06em;'
        'flex-shrink:0;">CVF</div>'
        '<div style="line-height:1.25;">'
        '<strong style="display:block;font-size:0.88rem;font-weight:700;'
        'color:#F1F5F9;font-family:Pretendard Variable,sans-serif;">'
        'Commute Value Finder</strong>'
        '<span style="font-size:0.65rem;color:#64748B;font-weight:400;'
        'letter-spacing:0.03em;">통근거리 기반 부동산 가치 분석</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="font-size:0.62rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#64748B;margin:1rem 0 0.5rem;">기준 위치</p>',
        unsafe_allow_html=True,
    )
    company_input = st.text_input("회사 위치", value=COMPANY_NAME, label_visibility="collapsed")
    if company_input == COMPANY_NAME:
        company_lat, company_lng = COMPANY_LAT, COMPANY_LNG
    else:
        coords = geocode_address(company_input)
        if coords:
            company_lat, company_lng = coords
            st.success("위치 반영됨")
        else:
            company_lat, company_lng = COMPANY_LAT, COMPANY_LNG
            st.warning("주소를 찾을 수 없어 기본 위치 사용")
    st.caption("통근 시간 데이터는 기존 분석 기준")

    st.divider()
    st.markdown(
        '<p style="font-size:0.62rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#64748B;margin:0.5rem 0 0.5rem;">필터 조건</p>',
        unsafe_allow_html=True,
    )
    max_commute = st.slider("통근 시간 상한 (분)", 15, 90, 60)
    max_price = st.slider("평당가 상한 (만원/㎡)", 500, 5000, 3000, step=100)

    st.divider()
    st.markdown(
        '<p style="font-size:0.62rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#64748B;margin:0.5rem 0 0.5rem;">Zone 필터</p>',
        unsafe_allow_html=True,
    )
    show_blue = st.checkbox("Blue Zone — 저평가", value=True)
    show_gray = st.checkbox("Gray Zone — 적정가", value=True)
    show_red = st.checkbox("Red Zone — 프리미엄", value=True)

    st.divider()
    show_subway = st.toggle("지하철 노선 표시", value=False)

    st.divider()
    st.markdown(
        '<p style="font-size:0.62rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#64748B;margin:0.5rem 0 0.5rem;">AI 브리핑</p>',
        unsafe_allow_html=True,
    )
    budget_input = st.text_input("예산", value="전세 5억")
    priority_input = st.text_input("우선순위", value="학군보다 통근 중요")
    briefing_btn = st.button("AI 브리핑 생성", use_container_width=True)


# ────────────────────────────────────────────────────────
# 데이터 로드 & 필터링
# ────────────────────────────────────────────────────────

dong_all = load_dong_data()
raw_all = load_raw_data()

if dong_all.empty:
    st.error("데이터가 없습니다. `python main.py`를 먼저 실행하세요.")
    st.stop()

apt_lookup = build_apt_lookup(raw_all) if not raw_all.empty else {}

zones = []
if show_blue: zones.append("Blue")
if show_gray: zones.append("Gray")
if show_red:  zones.append("Red")

if not zones:
    st.warning("Zone을 1개 이상 선택하세요.")
    st.stop()

filtered = dong_all[
    (dong_all["commute_minutes"] <= max_commute)
    & (dong_all["avg_price_per_sqm"] <= max_price)
    & (dong_all["zone"].isin(zones))
].copy()

blue_f = filtered[filtered["zone"] == "Blue"]


# ────────────────────────────────────────────────────────
# 메인 콘텐츠
# ────────────────────────────────────────────────────────

# Header (인라인 스타일)
st.markdown(
    '<div style="padding:0 0 1.2rem;margin-bottom:1.5rem;'
    'border-bottom:1px solid #1F2A3D;position:relative;">'
    '<div style="position:absolute;bottom:-1px;left:0;width:100px;height:2px;'
    'background:linear-gradient(90deg,#3B82F6,#06B6D4);border-radius:1px;"></div>'
    '<h1 style="font-size:1.35rem;font-weight:700;letter-spacing:-0.01em;'
    'color:#F1F5F9;margin:0 0 2px;padding:0;line-height:1.3;'
    'font-family:Pretendard Variable,Pretendard,sans-serif;">Commute Value Finder</h1>'
    '<p style="font-size:0.78rem;color:#64748B;margin:0;font-weight:400;'
    'letter-spacing:0.02em;font-family:Pretendard Variable,Pretendard,sans-serif;">'
    '통근거리 기반 부동산 가치 분석 플랫폼</p></div>',
    unsafe_allow_html=True,
)

# Metric Cards (st.columns 레이아웃 + 인라인 스타일)
blue_avg_price = f"{blue_f['avg_price_per_sqm'].mean():,.0f}" if len(blue_f) else "—"
blue_avg_commute = f"{blue_f['commute_minutes'].mean():.0f}" if len(blue_f) else "—"

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    metric_card("분석 동 수", str(len(filtered)), "개",
                f"전체 {len(dong_all)}개 중", "#3B82F6")
with mc2:
    metric_card("Blue Zone", str(len(blue_f)), "개",
                "저평가 지역", "#60A5FA")
with mc3:
    metric_card("Blue 평균 평당가", blue_avg_price, "만원/㎡",
                accent="#06B6D4")
with mc4:
    metric_card("Blue 평균 통근", blue_avg_commute, "분",
                accent="#34D399")


# ────────────────────────────────────────────────────────
# 탭
# ────────────────────────────────────────────────────────

tab_map, tab_data, tab_brief, tab_reg = st.tabs(
    ["지도", "분석 결과", "AI 브리핑", "회귀 분석"])

# ── 탭 1: 지도 ──
with tab_map:
    if filtered.empty:
        st.warning("필터 조건에 맞는 동이 없습니다.")
    else:
        fmap = build_map(filtered, apt_lookup, show_subway,
                         company_lat, company_lng, company_input)
        components.html(fmap._repr_html_(), height=680)
        st.caption("마커를 클릭하면 아파트 거래 내역이 팝업으로 표시됩니다.")

# ── 탭 2: 분석 결과 ──
with tab_data:
    st.subheader("동별 분석 데이터")

    COL_MAP = {
        "구": "구", "법정동": "법정동", "zone": "Zone",
        "commute_minutes": "통근(분)", "avg_price_per_sqm": "평당가(만원/㎡)",
        "median_price": "중위거래가(만원)", "residual": "잔차",
        "deviation_pct": "편차(%)", "transaction_count": "거래건수",
    }
    disp_cols = [c for c in COL_MAP if c in filtered.columns]
    disp = filtered[disp_cols].rename(columns=COL_MAP)

    st.dataframe(disp.sort_values("잔차"), use_container_width=True, height=400)

    csv = disp.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV 다운로드", data=csv,
                       file_name=f"commute_value_analysis_{date.today()}.csv",
                       mime="text/csv")

    st.divider()
    st.subheader("아파트 상세 보기")
    if not raw_all.empty and not filtered.empty:
        options = sorted((filtered["구"] + " " + filtered["법정동"]).unique())
        sel = st.selectbox("동 선택", options, index=None, placeholder="동을 선택하세요...")
        if sel:
            gu, dn = sel.split(" ", 1)
            apts = raw_all[
                (raw_all["구"] == gu) & (raw_all["법정동"] == dn)
            ][["아파트명", "건축년도", "전용면적", "거래금액", "층", "거래년월"]].copy()
            st.write(f"**{sel}** ({len(apts)}건)")
            st.dataframe(apts.sort_values("거래년월", ascending=False),
                         use_container_width=True, height=300)

# ── 탭 3: AI 브리핑 ──
with tab_brief:
    if briefing_btn:
        with st.spinner("AI 브리핑 생성 중..."):
            st.session_state["briefing"] = run_briefing(
                dong_all, budget_input, max_commute, priority_input, company_input)

    if "briefing" in st.session_state:
        st.markdown(
            f'<div style="background:#111827;border:1px solid #1F2A3D;'
            f'border-radius:10px;padding:24px 28px;line-height:1.85;'
            f'color:#94A3B8;font-size:0.88rem;'
            f'font-family:Pretendard Variable,Pretendard,sans-serif;">'
            f'{st.session_state["briefing"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("사이드바에서 **AI 브리핑 생성** 버튼을 클릭하세요.")

# ── 탭 4: 회귀 분석 ──
with tab_reg:
    st.subheader("통근 시간 vs 평당가 — 선형 회귀")

    plot_path = ROOT / "output" / "regression_plot.png"
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
    else:
        st.warning("`python src/zone_analyzer.py`를 먼저 실행하세요.")

    if "commute_minutes" in dong_all.columns:
        from sklearn.linear_model import LinearRegression
        try:
            clean = dong_all.dropna(subset=["commute_minutes", "avg_price_per_sqm"])
            model = LinearRegression().fit(
                clean[["commute_minutes"]], clean["avg_price_per_sqm"])
            r2 = model.score(clean[["commute_minutes"]], clean["avg_price_per_sqm"])
            coef = model.coef_[0]
        except Exception as e:
            st.error(f"회귀 분석 오류: {e}")
            st.stop()

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(
                '<div style="background:#111827;border:1px solid #1F2A3D;'
                'border-radius:10px;padding:16px 18px;text-align:center;">'
                '<div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;'
                'letter-spacing:0.05em;margin-bottom:4px;">R² (결정계수)</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#F1F5F9;">'
                f'{r2:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                '<div style="background:#111827;border:1px solid #1F2A3D;'
                'border-radius:10px;padding:16px 18px;text-align:center;">'
                '<div style="font-size:0.68rem;color:#64748B;text-transform:uppercase;'
                'letter-spacing:0.05em;margin-bottom:4px;">통근 1분 증가 시</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#F1F5F9;">'
                f'{coef:+,.1f}'
                f'<span style="font-size:0.7rem;color:#94A3B8;"> 만원/㎡</span></div></div>',
                unsafe_allow_html=True,
            )

        if r2 >= 0.5:
            st.success("모델 설명력 양호 — 통근 시간이 평당가 변동의 50% 이상 설명")
        elif r2 >= 0.3:
            st.info("모델 설명력 보통 — 통근 외 요인(학군, 인프라 등)도 상당한 영향")
        else:
            st.warning("모델 설명력 낮음 — 통근 외 요인이 가격에 더 큰 영향을 미침")
