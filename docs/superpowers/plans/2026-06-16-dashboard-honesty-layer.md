# Dashboard + Honesty Layer Implementation Plan (Plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 단지 단위 산출물을 사용자가 오해 없이·행동 가능하게 받도록, 정직성 레이어(후보 재프레이밍·리스크 플래그·불확실성·백테스트 한계 명시)와 행동 레이어(랭킹·외부 링크아웃·교통호재 오버레이)를 갖춘 Streamlit 대시보드로 마감한다.

**Architecture:** 순수 로직(`transit_projects.py`, `dashboard_logic.py`)을 TDD로 분리하고, `app.py`를 단지 단위(`complex_zones.csv`) + 새 `briefing.py` 기반으로 재구성한다. UI는 앱을 실제 실행해 검증한다.

**Tech Stack:** Python 3.14, pandas, streamlit, folium, openai, pytest.

---

## Environment Notes (구현자 필독)
- Work from `commute_value_finder/`. Python launcher is **`py`**. Run tests `py -m pytest ...`.
- git identity 설정됨. Task 0에서 브랜치 `feat/dashboard-honesty` 생성·사용.
- 산출물: `data/complex_zones.csv` (컬럼: 구·법정동·아파트명·n·last_ym·입지가치지수·commute_minutes·subway_dist_km·nearest_station·final_resid·deviation_pct·zone·confidence·blue_candidate_lowconf·lat·lon·avg_price_per_sqm·median_amount).
- 모듈: `src/briefing.py`(build_context/get_briefing/rule_based_briefing), `src/validation.py`.
- **Plan 3 발견(대시보드에 반영):** 백테스트 `signal_valid=False`(Blue가 미래 상승을 예측 못함 — 스크리닝 용도), Moran's I=0.57 p=0.005(유의한 공간 자기상관), 상위 Blue에 극단 편차(예: -76%) 존재(이상치 의심).
- streamlit/folium 미설치 시 설치: `py -m pip install streamlit folium streamlit-folium`.

---

## File Structure

| 파일 | 책임 |
|---|---|
| `data/transit_projects.json` (신규) | GTX·신설노선 예정역 정적 데이터 |
| `src/transit_projects.py` (신규) | 호재역 로드 + 최근접 호재 거리 |
| `src/dashboard_logic.py` (신규) | 리스크 플래그·랭킹·링크아웃·라벨 (순수함수, TDD) |
| `app.py` (재구성) | 단지 단위 대시보드 + 정직성/행동 레이어 |
| `tests/` | 신규 단위 테스트 |

---

## Task 0: 브랜치 생성
- [ ] **Step 1:**
```bash
cd "C:/Users/ubumi/Desktop/2. 코딩연습/0. 2026/cursor/0401_01_commute"
git checkout main && git checkout -b feat/dashboard-honesty
git branch --show-current
```
Expected: `feat/dashboard-honesty`

---

## Task 1: 교통호재 데이터 + 모듈

**Files:**
- Create: `commute_value_finder/data/transit_projects.json`
- Create: `commute_value_finder/src/transit_projects.py`
- Test: `commute_value_finder/tests/test_transit_projects.py`

- [ ] **Step 1: Create `data/transit_projects.json`** (GTX·신설노선 예정역, 서울/인접 주요역 — 좌표는 근사값, "예정"):
```json
[
  {"name": "GTX-A 삼성", "line": "GTX-A", "lat": 37.5088, "lon": 127.0631, "open_year": 2028},
  {"name": "GTX-A 수서", "line": "GTX-A", "lat": 37.4870, "lon": 127.1010, "open_year": 2024},
  {"name": "GTX-A 서울역", "line": "GTX-A", "lat": 37.5547, "lon": 126.9707, "open_year": 2026},
  {"name": "GTX-A 연신내", "line": "GTX-A", "lat": 37.6191, "lon": 126.9211, "open_year": 2026},
  {"name": "GTX-C 청량리", "line": "GTX-C", "lat": 37.5800, "lon": 127.0470, "open_year": 2028},
  {"name": "GTX-C 광운대", "line": "GTX-C", "lat": 37.6230, "lon": 127.0610, "open_year": 2028},
  {"name": "GTX-C 창동", "line": "GTX-C", "lat": 37.6530, "lon": 127.0477, "open_year": 2028},
  {"name": "GTX-C 양재", "line": "GTX-C", "lat": 37.4843, "lon": 127.0342, "open_year": 2028},
  {"name": "GTX-B 여의도", "line": "GTX-B", "lat": 37.5215, "lon": 126.9243, "open_year": 2030},
  {"name": "GTX-B 청량리", "line": "GTX-B", "lat": 37.5800, "lon": 127.0470, "open_year": 2030},
  {"name": "신안산선 영등포", "line": "신안산선", "lat": 37.5157, "lon": 126.9075, "open_year": 2025},
  {"name": "동북선 왕십리", "line": "동북선", "lat": 37.5614, "lon": 127.0378, "open_year": 2026},
  {"name": "동북선 상계", "line": "동북선", "lat": 37.6542, "lon": 127.0568, "open_year": 2026}
]
```

- [ ] **Step 2: Write test** — create `tests/test_transit_projects.py`:
```python
from src.transit_projects import load_projects, nearest_project_km


def test_load_projects_real_file():
    projects = load_projects()  # 기본 경로 data/transit_projects.json
    assert len(projects) >= 10
    assert all("lat" in p and "line" in p for p in projects)


def test_nearest_project_km():
    projects = [
        {"name": "GTX-A 삼성", "line": "GTX-A", "lat": 37.5088, "lon": 127.0631, "open_year": 2028},
        {"name": "GTX-C 창동", "line": "GTX-C", "lat": 37.6530, "lon": 127.0477, "open_year": 2028},
    ]
    dist, name, year = nearest_project_km(37.5100, 127.0640, projects)  # 삼성 근처
    assert name == "GTX-A 삼성"
    assert dist < 1.0
    assert year == 2028


def test_nearest_project_empty():
    dist, name, year = nearest_project_km(37.5, 127.0, [])
    assert name is None
```

- [ ] **Step 3: Implement** — create `src/transit_projects.py`:
```python
"""교통 호재(GTX·신설노선) 예정역 — 모델 피처가 아니라 Blue 후보 위 설명 레이어."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subway_access import haversine_km

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "transit_projects.json"


def load_projects(path=None):
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


def nearest_project_km(lat, lon, projects):
    """최근접 호재역까지 거리(km), 역명, 개통예정연도. 없으면 (inf, None, None)."""
    best = (float("inf"), None, None)
    for p in projects:
        d = haversine_km(lat, lon, p["lat"], p["lon"])
        if d < best[0]:
            best = (d, p["name"], p.get("open_year"))
    return best
```

- [ ] **Step 4: Run** `py -m pytest tests/test_transit_projects.py -v` → 3 PASS. Debug if needed.

- [ ] **Step 5: Commit**
```bash
git add data/transit_projects.json src/transit_projects.py tests/test_transit_projects.py
git commit -m "feat: add transit-project (GTX/new-line) overlay data and nearest lookup"
```

---

## Task 2: `dashboard_logic.py` — 리스크·랭킹·링크아웃·라벨

**Files:**
- Create: `commute_value_finder/src/dashboard_logic.py`
- Test: `commute_value_finder/tests/test_dashboard_logic.py`

- [ ] **Step 1: Write test** — create `tests/test_dashboard_logic.py`:
```python
import pandas as pd
from src.dashboard_logic import (
    zone_label, risk_flags, naver_land_url, hogangnono_url, rank_candidates,
)


def test_zone_label():
    assert zone_label("Blue") == "저평가 후보 (확인 필요)"
    assert zone_label("Gray") == "적정"
    assert zone_label("Red") == "프리미엄"


def test_risk_flags():
    base = {"n": 12, "confidence": "높음", "deviation_pct": -20.0, "subway_dist_km": 0.3}
    assert risk_flags(base) == []
    # 극단 편차
    assert any("극단" in f for f in risk_flags({**base, "deviation_pct": -70.0}))
    # 거래 적음
    assert any("거래" in f for f in risk_flags({**base, "n": 3}))
    # 역 멀다
    assert any("역" in f for f in risk_flags({**base, "subway_dist_km": 1.5}))


def test_linkout_urls():
    n = naver_land_url("노원구", "상계동", "A아파트")
    h = hogangnono_url("A아파트")
    assert n.startswith("https://") and "A아파트" in n
    assert h.startswith("https://") and "A아파트" in h


def test_rank_candidates_filters_and_sorts():
    zones = pd.DataFrame({
        "구": ["노원구", "도봉구", "강남구"],
        "법정동": ["상계동", "창동", "삼성동"],
        "아파트명": ["A", "B", "C"],
        "zone": ["Blue", "Blue", "Gray"],
        "commute_minutes": [40, 70, 20],
        "avg_price_per_sqm": [1500.0, 1400.0, 5000.0],
        "final_resid": [-0.2, -0.4, 0.0],
    })
    out = rank_candidates(zones, max_commute=60, max_price=3000, top_n=10)
    # Gray(강남) 제외, 통근 70분(도봉) 제외 → 노원 A만
    assert list(out["아파트명"]) == ["A"]
```

- [ ] **Step 2: Run** `py -m pytest tests/test_dashboard_logic.py -v` → FAIL.

- [ ] **Step 3: Implement** — create `src/dashboard_logic.py`:
```python
"""대시보드 표시 로직: 라벨·리스크 플래그·외부 링크·후보 랭킹 (순수함수)."""
from urllib.parse import quote

ZONE_LABELS = {
    "Blue": "저평가 후보 (확인 필요)",
    "Gray": "적정",
    "Red": "프리미엄",
}


def zone_label(zone: str) -> str:
    return ZONE_LABELS.get(zone, zone)


def risk_flags(row) -> list:
    """단지의 '왜 싸 보이는지' 자동 리스크 플래그."""
    flags = []
    dev = row.get("deviation_pct")
    if dev is not None and abs(dev) > 50:
        flags.append("극단 편차 — 이상치/데이터 확인 필요")
    n = row.get("n")
    if n is not None and n < 5:
        flags.append("거래 적음 — 표본 신중")
    if row.get("confidence") == "낮음":
        flags.append("표본 신뢰 낮음")
    sd = row.get("subway_dist_km")
    if sd is not None and sd > 1.0:
        flags.append("역세권 아님 — 도보 먼 편")
    return flags


def naver_land_url(gu: str, dong: str, apt: str) -> str:
    """네이버 부동산 통합검색 링크 (현재 호가 확인용)."""
    return "https://m.land.naver.com/search/result/" + quote(f"{dong} {apt}")


def hogangnono_url(apt: str) -> str:
    """호갱노노 검색 링크."""
    return "https://hogangnono.com/search?q=" + quote(apt)


def rank_candidates(zones, max_commute, max_price, top_n=20):
    """Blue 후보를 조건으로 필터 → 저평가순(잔차 오름차순) 정렬."""
    df = zones[
        (zones["zone"] == "Blue")
        & (zones["commute_minutes"] <= max_commute)
        & (zones["avg_price_per_sqm"] <= max_price)
    ].copy()
    return df.sort_values("final_resid").head(top_n)
```

- [ ] **Step 4: Run** `py -m pytest tests/test_dashboard_logic.py -v` → 4 PASS. Debug until green.

- [ ] **Step 5: Commit**
```bash
git add src/dashboard_logic.py tests/test_dashboard_logic.py
git commit -m "feat: add dashboard logic (risk flags, ranking, linkouts, labels)"
```

---

## Task 3: `app.py` — 단지 단위 대시보드 + 정직성 레이어

**Files:**
- Modify (full replace): `commute_value_finder/app.py`

> 기존 app.py는 동 단위(`dong_with_commute.csv`) + 옛 LLM을 쓴다. 단지 단위(`complex_zones.csv`) + 새 `briefing.py` + 정직성/행동 레이어로 재구성한다.

- [ ] **Step 1: REPLACE the ENTIRE contents of `commute_value_finder/app.py` with:**
```python
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


# ── 데이터 ──
zones = load_zones()
if zones.empty:
    st.error("데이터가 없습니다. `py run_model.py`를 먼저 실행하세요.")
    st.stop()

# ── 헤더 + 정직성 배너 ──
st.title("🏠 Commute-Value Finder")
st.caption("통근·지하철 접근성 대비 가격이 낮은 단지를 '확인 필요 후보'로 스크리닝합니다.")
st.warning(
    "**이 도구는 예측기가 아니라 스크리너입니다.** 백테스트 결과, 저평가 후보가 "
    "향후 가격 상승을 예측하지는 못했습니다(통근·지하철 대비 '지속적으로 싼' 곳을 찾는 용도). "
    "또한 잔차에 공간 자기상관이 유의(Moran's I≈0.57)해, 학군 등 누락 요인의 영향이 섞여 "
    "있을 수 있습니다. 후보는 반드시 **직접 확인**하세요."
)

# ── 사이드바 필터 ──
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

# ── 지표 ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("필터된 단지", f"{len(filtered):,}")
c2.metric("Blue 후보", f"{int((zones['zone']=='Blue').sum()):,}")
c3.metric("Blue 평균 평당가",
          f"{blue_f['avg_price_per_sqm'].mean():,.0f}" if len(blue_f) else "—")
c4.metric("Blue 평균 통근",
          f"{blue_f['commute_minutes'].mean():.0f}분" if len(blue_f) else "—")

tab_map, tab_rank, tab_brief, tab_model = st.tabs(
    ["지도", "후보 랭킹", "AI 브리핑", "모델·검증"])

# ── 탭1: 지도 ──
with tab_map:
    m = folium.Map(location=[37.5665, 126.978], zoom_start=11, tiles="cartodbpositron")
    # 성능: 마커가 많으면 클러스터
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

# ── 탭2: 후보 랭킹 (행동 레이어) ──
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

# ── 탭3: AI 브리핑 ──
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

# ── 탭4: 모델·검증 ──
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
```

- [ ] **Step 2: Syntax check** — `py -c "import ast; ast.parse(open('app.py',encoding='utf-8').read()); print('app.py parses OK')"` → `app.py parses OK`.

- [ ] **Step 3: Run full suite** `py -m pytest tests/ -q` → ALL pass (app.py has no unit tests; this confirms no import breakage in modules).

- [ ] **Step 4: Commit**
```bash
git add app.py
git commit -m "feat: rebuild dashboard on complex data with honesty and action layers"
```

---

## Task 4: 앱 실행 검증 (스모크 + 스크린샷)

**Files:** none (verification only)

- [ ] **Step 1: 의존성 확인/설치**
```bash
py -c "import streamlit, folium" 2>/dev/null || py -m pip install streamlit folium
```

- [ ] **Step 2: 앱 import 스모크 (Streamlit 런타임 없이 모듈 적재 확인)**
Run: `py -c "import ast,sys; sys.path.insert(0,'.'); ast.parse(open('app.py',encoding='utf-8').read()); from src import dashboard_logic, transit_projects, briefing; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 3: Streamlit 헤드리스 기동 스모크.** 백그라운드로 띄우고 응답을 확인:
```bash
py -m streamlit run app.py --server.headless true --server.port 8765 &
sleep 12
py -c "import urllib.request; print('HTTP', urllib.request.urlopen('http://localhost:8765', timeout=10).status)"
```
Expected: `HTTP 200`. (확인 후 프로세스 종료: `pkill -f 'streamlit run' || true`)
If port busy, use 8766. If `streamlit` CLI not found, run `py -m streamlit ...`.

- [ ] **Step 4 (선택): 스크린샷.** webapp-testing 스킬 또는 Playwright가 가능하면 `http://localhost:8765`를 캡처해 지도/랭킹/배너가 렌더되는지 육안 확인. 불가하면 Step 3의 HTTP 200으로 갈음.

- [ ] **Step 5: 검증 결과 보고.** HTTP 200 여부, import OK 여부, (가능하면) 스크린샷 관찰을 보고. 코드 커밋은 Task 3에서 끝났으므로 별도 커밋 없음(스크린샷 파일은 커밋하지 말 것 — gitignore 대상).

---

## Self-Review (작성자 점검)
**스펙 커버리지 (Plan 4 = §5.6/§5.8/§5.9):**
- §5.9 교통호재 정적데이터 + 오버레이 + 거리 → Task 1, app 지도 ✓
- §5.8 재프레이밍·리스크 플래그·불확실성·랭킹·링크아웃 → Task 2, app 랭킹탭 ✓
- §5.6 단지 마커(클러스터)·다변량 모델 노출 → Task 3 ✓
- Plan 3 발견(백테스트 한계·Moran's I caveat) 반영 → app 배너+모델탭 ✓
- 새 briefing.py 전환 → app 브리핑탭 ✓

**범위 밖:** 옛 `llm_briefing.py`/`zone_analyzer.py` 정리(미사용이 됨 — 삭제는 별도 정리 PR), 실시간 호가 크롤링(링크아웃으로 대체), 출근방식 선택(Phase B).

**타입:** app은 complex_zones.csv 컬럼만 사용. 순수 로직은 모두 TDD. UI는 기동 스모크로 검증.

**플레이스홀더:** 없음.

---
## Execution Handoff
Plan 4 실행 후 전체 프로젝트(Plan 1~4) 완료. 최종 정리(미사용 모듈, README)는 후속으로 제안.
