"""Claude API 기반 LLM 브리핑 모듈 (Phase 3)

Step 1: 사용자 조건 입력 (예산, 통근, 우선순위)
Step 2: Blue Zone 데이터 → LLM 컨텍스트 생성
Step 3: Claude API 호출 (실패 시 규칙 기반 폴백)
Step 4: 결과 출력 / 저장 / map_final.html 생성
"""

import sys
import re
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import folium
from sklearn.linear_model import LinearRegression

from config import COMPANY_LAT, COMPANY_LNG, COMPANY_NAME
from src.utils import get_project_root
from src.zone_analyzer import make_zone_map


# ────────────────────────────────────────────────────────
# Step 1: 사용자 입력
# ────────────────────────────────────────────────────────

def _parse_budget(text: str) -> tuple[str, int]:
    """'전세 5억' → ('전세', 50000만원)"""
    text = text.strip()
    deal_type = "전세" if "전세" in text else "매매"
    num_text = re.sub(r"(전세|매매|이하|이내|원|약)", "", text).strip()

    amount = 0
    if "억" in num_text:
        parts = num_text.split("억")
        amount = int(float(parts[0].strip() or 0)) * 10000
        rest = parts[1].replace("천", "000").replace("만", "").strip()
        if rest:
            amount += int(float(rest))
    elif num_text:
        amount = int(float(num_text.replace("만", "")))

    return deal_type, amount


def get_user_preferences() -> dict:
    """터미널에서 예산 / 통근 상한 / 우선순위를 입력받는다."""
    print("\n--- 조건 입력 ---")

    raw = input("1) 예산 (예: '전세 5억', '매매 8억'): ").strip()
    if not raw:
        raw = "전세 50000"
    budget_type, budget_amount = _parse_budget(raw)
    print(f"   → {budget_type} {budget_amount:,}만원")

    commute_raw = input("2) 최대 통근 시간 (분, 기본 45): ").strip()
    max_commute = int(commute_raw) if commute_raw.isdigit() else 45
    print(f"   → {max_commute}분")

    print("3) 우선순위 (한 줄씩 입력, 빈 줄로 종료):")
    priorities = []
    while True:
        line = input("   > ").strip()
        if not line:
            break
        priorities.append(line)
    if not priorities:
        priorities = ["특별한 선호 없음"]
    print(f"   → {', '.join(priorities)}")

    return {
        "budget_type": budget_type,
        "budget_amount": budget_amount,
        "max_commute": max_commute,
        "priorities": priorities,
    }


# ────────────────────────────────────────────────────────
# Step 2: 데이터 컨텍스트 빌더
# ────────────────────────────────────────────────────────

def build_context(dong: pd.DataFrame, prefs: dict,
                   company_name: str | None = None) -> tuple[str, pd.DataFrame]:
    """사용자 조건에 맞는 Blue Zone 필터 → LLM 컨텍스트 문자열 + 후보 DF"""
    blue = dong[dong["zone"] == "Blue"].copy()
    blue = blue[blue["commute_minutes"] <= prefs["max_commute"]]
    # 잔차 오름차순 = 가장 저평가된 동이 먼저
    blue = blue.sort_values("residual", ascending=True)
    candidates = blue.head(15)

    if candidates.empty:
        # 조건 완화: Gray 포함
        pool = dong[dong["commute_minutes"] <= prefs["max_commute"]]
        candidates = pool.sort_values("residual").head(15)

    avg_all = dong["avg_price_per_sqm"].mean()
    from config import get_analysis_months
    months = get_analysis_months()
    date_range = f"{months[0][:4]}.{months[0][4:]}~{months[-1][:4]}.{months[-1][4:]}"

    co_name = company_name or COMPANY_NAME
    lines = [f"데이터 기준: {date_range}",
             f"서울 전체 평균 평당가: {avg_all:,.0f}만원/㎡",
             f"기준 회사: {co_name}", ""]
    lines.append("Blue Zone 후보 (저평가순):")
    for i, (_, r) in enumerate(candidates.iterrows(), 1):
        lines.append(
            f"{i}. {r['구']} {r['법정동']} | "
            f"통근 {int(r['commute_minutes'])}분 | "
            f"평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡ | "
            f"잔차 {r['residual']:+,.0f} | "
            f"중위가 {r['median_price']:,.0f}만원 | "
            f"{int(r['transaction_count'])}건"
        )

    context = "\n".join(lines)
    # 토큰 절약: 1000자 제한
    if len(context) > 1000:
        context = context[:997] + "..."

    return context, candidates


# ────────────────────────────────────────────────────────
# Step 3: OpenAI API 브리핑
# ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "당신은 서울 부동산 데이터 분석가입니다.\n"
    "반드시 아래 데이터에 있는 지역만 추천하세요. "
    "데이터에 없는 지역은 절대 언급하지 마세요.\n"
    "추천 형식: 동네명 → 추천 이유 (통근, 가격, 특징) → 주의사항 순으로 작성하세요."
)

USER_TEMPLATE = (
    "[분석 데이터]\n{context}\n\n"
    "[사용자 조건]\n예산: {budget}\n통근 상한: {commute}분\n"
    "우선순위: {priority}\n\n"
    "위 데이터 기반으로 최적의 동네 3곳을 추천하고 "
    "각각의 이유를 구체적으로 설명해주세요. "
    "마지막에 다음 안내를 추가해주세요: "
    "'※ 이 분석은 개인 주거지 탐색용입니다. "
    "거점오피스 최적 입지 분석은 직원 거주지 데이터가 추가로 필요합니다.'"
)


def get_llm_briefing(context: str, prefs: dict, api_key: str) -> str:
    """OpenAI API로 브리핑 생성 (1회 재시도, 실패 시 빈 문자열)"""
    from openai import OpenAI
    from config import LLM_MODEL, LLM_MAX_TOKENS

    budget_str = f"{prefs['budget_type']} {prefs['budget_amount']:,}만원 이내"
    user_msg = USER_TEMPLATE.format(
        context=context,
        budget=budget_str,
        commute=prefs["max_commute"],
        priority=", ".join(prefs["priorities"]),
    )

    client = OpenAI(api_key=api_key)

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 0:
                print(f"  [WARN] API 오류, 재시도 중... ({e})")
            else:
                print(f"  [ERROR] API 2회 실패: {e}")
    return ""


def _fallback_briefing(candidates: pd.DataFrame, prefs: dict) -> str:
    """API 실패 시 규칙 기반 추천"""
    top3 = candidates.head(3)
    lines = ["[규칙 기반 추천 — Claude API 미사용]\n"]

    for i, (_, r) in enumerate(top3.iterrows(), 1):
        lines.append(
            f"{i}. {r['구']} {r['법정동']}\n"
            f"   통근 {int(r['commute_minutes'])}분 | "
            f"평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡ | "
            f"주변 대비 {r['deviation_pct']:+.1f}%\n"
            f"   → 통근 시간 대비 가격이 저평가된 지역입니다. "
            f"거래 {int(r['transaction_count'])}건으로 "
            f"{'유동성 충분' if r['transaction_count'] >= 30 else '거래량 확인 필요'}.\n"
        )

    lines.append(
        "\n※ 이 분석은 개인 주거지 탐색용입니다. "
        "거점오피스 최적 입지 분석은 직원 거주지 데이터가 추가로 필요합니다."
    )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────
# Step 4: 결과 출력 / 저장 / 최종 지도
# ────────────────────────────────────────────────────────

def _extract_recommended(text: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """브리핑에서 언급된 동 추출 (최대 3개)"""
    found = []
    for _, r in candidates.iterrows():
        if r["법정동"] in text or f"{r['구']} {r['법정동']}" in text:
            found.append(r)
            if len(found) >= 3:
                break
    if not found:
        found = [r for _, r in candidates.head(3).iterrows()]
    return pd.DataFrame(found)


def display_and_save(briefing: str, dong: pd.DataFrame,
                     candidates: pd.DataFrame, prefs: dict):
    """터미널 출력 + 파일 저장 + map_final.html 생성"""
    output_dir = get_project_root() / "output"
    output_dir.mkdir(exist_ok=True)

    budget_str = f"{prefs['budget_type']} {prefs['budget_amount']:,}만원"

    # ── 터미널 출력 ──
    print(f"\n{'═'*55}")
    print(f"🏠 Commute-Value Finder — AI 브리핑")
    print(f"{'═'*55}")
    print(f"📋 조건: {budget_str} / 통근 {prefs['max_commute']}분 / "
          f"{', '.join(prefs['priorities'])}")
    print(f"{'─'*55}")
    print(briefing)
    print(f"{'─'*55}")

    # ── 텍스트 저장 ──
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt_path = output_dir / "briefing_result.txt"
    txt_path.write_text(
        f"Commute-Value Finder 브리핑 ({ts})\n"
        f"조건: {budget_str} / 통근 {prefs['max_commute']}분 / "
        f"{', '.join(prefs['priorities'])}\n"
        f"{'='*55}\n\n{briefing}\n",
        encoding="utf-8",
    )
    print(f"\n💾 저장: {txt_path}")

    # ── map_final.html ──
    recommended = _extract_recommended(briefing, candidates)
    fmap = make_zone_map(dong)

    # 추천 동 별 마커
    rec_fg = folium.FeatureGroup(name="추천 지역", show=True)
    for _, r in recommended.iterrows():
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
            tooltip=f"⭐ 추천: {r['구']} {r['법정동']}",
            popup=folium.Popup(
                f"<b>⭐ 추천 지역</b><br>"
                f"{r['구']} {r['법정동']}<br>"
                f"통근 {int(r['commute_minutes'])}분 | "
                f"평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡",
                max_width=250,
            ),
        ).add_to(rec_fg)
    rec_fg.add_to(fmap)

    # 브리핑 사이드 패널
    safe_html = (
        briefing.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    panel_html = f"""
    <div style="position:fixed; top:10px; left:10px; z-index:1000;
         width:370px; max-height:85vh; overflow-y:auto;
         background:white; padding:15px 18px; border-radius:8px;
         border:1px solid #ddd; font-size:12.5px; line-height:1.7;
         box-shadow:2px 2px 8px rgba(0,0,0,.2);">
        <h3 style="margin:0 0 8px 0; font-size:15px;">
            🏠 AI 브리핑</h3>
        <div style="color:#555; font-size:11px; margin-bottom:8px;">
            조건: {budget_str} / 통근 {prefs['max_commute']}분</div>
        <div>{safe_html}</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(panel_html))

    folium.LayerControl(collapsed=False).add_to(fmap)

    map_path = output_dir / "map_final.html"
    fmap.save(str(map_path))
    print(f"🗺️  지도: {map_path}")


# ────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────

def _ensure_zone_columns(dong: pd.DataFrame) -> pd.DataFrame:
    """zone 컬럼이 없으면 재계산"""
    if "zone" in dong.columns:
        return dong
    model = LinearRegression().fit(dong[["commute_minutes"]], dong["avg_price_per_sqm"])
    pred = model.predict(dong[["commute_minutes"]].values)
    dong = dong.copy()
    dong["predicted_price"] = pred
    dong["residual"] = dong["avg_price_per_sqm"] - pred
    sigma = dong["residual"].std()
    dong["zone"] = "Gray"
    dong.loc[dong["residual"] < -sigma, "zone"] = "Blue"
    dong.loc[dong["residual"] > sigma, "zone"] = "Red"
    dong["deviation_pct"] = (dong["residual"] / dong["predicted_price"] * 100).round(1)
    return dong


def main(demo: bool = False):
    from dotenv import load_dotenv
    load_dotenv(get_project_root() / ".env")

    data_dir = get_project_root() / "data"
    commute_path = data_dir / "dong_with_commute.csv"

    if not commute_path.exists():
        print("[ERROR] 데이터 없음. zone_analyzer.py를 먼저 실행하세요.")
        return

    dong = pd.read_csv(commute_path)
    dong = _ensure_zone_columns(dong)
    print(f"데이터 로드: {len(dong)}개 동 "
          f"(Blue {(dong['zone']=='Blue').sum()} / "
          f"Gray {(dong['zone']=='Gray').sum()} / "
          f"Red {(dong['zone']=='Red').sum()})\n")

    # Step 1: 조건 입력
    if demo:
        prefs = {
            "budget_type": "전세",
            "budget_amount": 50000,
            "max_commute": 40,
            "priorities": ["학군보다 통근이 중요", "조용한 동네 선호"],
        }
        print(f"[DEMO] 조건: {prefs['budget_type']} {prefs['budget_amount']:,}만원 / "
              f"통근 {prefs['max_commute']}분 / {', '.join(prefs['priorities'])}")
    else:
        prefs = get_user_preferences()

    # Step 2: 컨텍스트
    print("\nStep 2: 컨텍스트 생성...")
    context, candidates = build_context(dong, prefs)
    print(f"  후보: {len(candidates)}개 동")

    # Step 3: LLM 브리핑
    print("\nStep 3: AI 브리핑 생성...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("여기에"):
        briefing = get_llm_briefing(context, prefs, api_key)
        if not briefing:
            print("  → API 실패, 규칙 기반 추천으로 대체")
            briefing = _fallback_briefing(candidates, prefs)
    else:
        print("  [INFO] OPENAI_API_KEY 미설정 → 규칙 기반 추천")
        briefing = _fallback_briefing(candidates, prefs)

    # Step 4: 출력 및 저장
    display_and_save(briefing, dong, candidates, prefs)


if __name__ == "__main__":
    demo_mode = "--demo" in sys.argv
    main(demo=demo_mode)
