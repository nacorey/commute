"""LLM 브리핑 (OpenAI gpt-5.4-mini, 구조화 JSON 출력).

제공된 단지 데이터에만 근거하도록 출력 스키마를 강제하고, 실패 시
규칙 기반으로 명시적으로 폴백한다.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DEFAULT_MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = (
    "당신은 서울 부동산 데이터 분석가입니다.\n"
    "규칙:\n"
    "1) 제공된 데이터에 있는 동/단지만 추천하세요. 없는 곳은 절대 언급 금지.\n"
    "2) 가격·통근·잔차 수치는 입력 데이터에서만 인용하세요.\n"
    "3) 거래건수가 적은 단지는 반드시 risk_notes에 표시하세요.\n"
    "4) 추천 이유는 가격/통근/거래신뢰도/주의로 나눠 쓰세요.\n"
    "5) 이 분석은 '저평가 확정'이 아니라 '확인이 필요한 후보'임을 disclaimer에 명시하세요."
)

BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "구": {"type": "string"},
                    "법정동": {"type": "string"},
                    "아파트명": {"type": "string"},
                    "reason": {
                        "type": "object",
                        "properties": {
                            "가격": {"type": "string"},
                            "통근": {"type": "string"},
                            "거래신뢰도": {"type": "string"},
                            "주의": {"type": "string"},
                        },
                        "required": ["가격", "통근", "거래신뢰도", "주의"],
                        "additionalProperties": False,
                    },
                    "avg_price_per_sqm": {"type": "number"},
                    "commute_min": {"type": "number"},
                    "residual": {"type": "number"},
                    "risk_notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["rank", "구", "법정동", "아파트명", "reason",
                             "avg_price_per_sqm", "commute_min", "residual", "risk_notes"],
                "additionalProperties": False,
            },
        },
        "disclaimer": {"type": "string"},
    },
    "required": ["recommendations", "disclaimer"],
    "additionalProperties": False,
}

DISCLAIMER = ("이 분석은 통근·지하철 대비 가격이 낮은 '확인 필요 후보'입니다. "
             "학군·향·소음·재건축 등은 직접 확인하세요. 개인 주거지 탐색용입니다.")


def build_context(zones: pd.DataFrame, prefs: dict):
    """Blue 후보를 조건으로 필터 → LLM 컨텍스트 문자열 + 후보 DF."""
    blue = zones[zones["zone"] == "Blue"].copy()
    blue = blue[blue["commute_minutes"] <= prefs["max_commute"]]
    blue = blue.sort_values("final_resid")  # 가장 저평가 먼저
    candidates = blue.head(10)

    lines = [f"기준: {prefs['budget_type']} {prefs['budget_amount']:,}만원 / "
             f"통근 {prefs['max_commute']}분 이내", "", "저평가 후보 단지:"]
    for i, (_, r) in enumerate(candidates.iterrows(), 1):
        lines.append(
            f"{i}. {r['구']} {r['법정동']} {r['아파트명']} | "
            f"통근 {int(r['commute_minutes'])}분 | "
            f"평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡ | "
            f"통근·지하철대비 {r['deviation_pct']:+.1f}% | "
            f"역거리 {r['subway_dist_km']:.2f}km | "
            f"거래 {int(r['n'])}건({r['confidence']})"
        )
    return "\n".join(lines), candidates


def _openai_caller_factory(api_key, model):
    def _call(model_name, system, user):
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_completion_tokens=1500,
            response_format={"type": "json_schema", "json_schema":
                             {"name": "briefing", "strict": True, "schema": BRIEFING_SCHEMA}},
        )
        return resp.choices[0].message.content
    return _call


def get_briefing(context: str, prefs: dict, api_key: str,
                 model: str = DEFAULT_MODEL, caller=None):
    """구조화 JSON 브리핑 생성. 실패 시 None(호출측이 규칙기반 폴백)."""
    caller = caller or _openai_caller_factory(api_key, model)
    user = (f"[분석 데이터]\n{context}\n\n"
            f"위 데이터의 단지 중 최적 3곳을 골라 구조화 JSON으로 추천하세요.")
    try:
        raw = caller(model, SYSTEM_PROMPT, user)
        data = json.loads(raw)
        if "recommendations" not in data:
            raise ValueError("missing recommendations")
        return data
    except Exception as e:
        print(f"[WARN] LLM 브리핑 실패 → 규칙기반 폴백 사용 (사유: {e})")
        return None


def rule_based_briefing(candidates: pd.DataFrame, prefs: dict) -> dict:
    """API 실패 시 규칙 기반 추천 (스키마 동일 형태)."""
    recs = []
    for i, (_, r) in enumerate(candidates.head(3).iterrows(), 1):
        risks = []
        if r["n"] < 5:
            risks.append("거래건수 부족 — 신중")
        if r.get("confidence") == "낮음":
            risks.append("표본 신뢰도 낮음")
        risks.append("학군·향·소음 등은 직접 확인 필요")
        recs.append({
            "rank": i, "구": r["구"], "법정동": r["법정동"], "아파트명": r["아파트명"],
            "reason": {
                "가격": f"평당가 {r['avg_price_per_sqm']:,.0f}만원/㎡ (통근·지하철 대비 {r['deviation_pct']:+.1f}%)",
                "통근": f"{int(r['commute_minutes'])}분",
                "거래신뢰도": f"{int(r['n'])}건 ({r.get('confidence','-')})",
                "주의": "모델 기준 저평가 '후보'이며 확정이 아님",
            },
            "avg_price_per_sqm": float(r["avg_price_per_sqm"]),
            "commute_min": float(r["commute_minutes"]),
            "residual": float(r["final_resid"]),
            "risk_notes": risks,
        })
    return {"recommendations": recs, "disclaimer": DISCLAIMER}
