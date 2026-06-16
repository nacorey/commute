# Validation + LLM Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 저평가 신호가 실제로 의미 있는지 백테스트·공간자기상관으로 자기검증하고, LLM 브리핑을 gpt-5.4-mini + 구조화 JSON으로 교체해 신뢰도를 끌어올린다. 동시에 단지 가격통계를 산출물에 추가해 LLM·대시보드가 실제 평당가를 쓸 수 있게 한다.

**Architecture:** `value_model.add_complex_price_stats`로 단지 평당가/중위가를 산출물에 더하고, `validation.py`(시간 홀드아웃 백테스트 + 수동 Moran's I), `briefing.py`(OpenAI 구조화 JSON 브리핑)를 신설한다. Moran's I는 PySAL 대신 numpy+sklearn KNN으로 직접 구현(Python 3.14 휠 리스크 회피).

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn, openai(2.41.1), pytest. (모두 설치 확인됨.)

---

## Environment Notes (구현자 필독)
- Work from `commute_value_finder/`. Python launcher is **`py`** (NOT python/python3). Run tests with `py -m pytest ...`.
- git identity 설정됨. Task 0에서 브랜치 `feat/validation-and-llm` 생성·사용.
- 보유 산출물: `data/complex_zones.csv`(Plan 2 결과; 컬럼 구·법정동·아파트명·n·last_ym·입지가치지수·commute_minutes·subway_dist_km·final_resid·deviation_pct·zone·confidence·blue_candidate_lowconf·lat·lon·nearest_station). `data/seoul_apt_transactions.csv`(원본). `.env`에 `OPENAI_API_KEY` 존재.
- Plan 1/2 모듈: `src/preprocessor.py`, `src/value_model.py`(fit_quality_model/aggregate_to_complex/classify_zones/ym_subtract_months), `src/commute.py`(load_dong_commute), `run_model.py`.
- **주의:** `value_model.aggregate_to_complex`의 기존 테스트는 `price_per_sqm`/`거래금액` 컬럼이 없는 합성 데이터를 쓴다. 그 함수는 건드리지 말 것(새 함수 add_complex_price_stats를 별도로 추가).

---

## File Structure

| 파일 | 책임 |
|---|---|
| `src/value_model.py` (수정) | `add_complex_price_stats` 추가 (단지 평당가·중위가) |
| `run_model.py` (수정) | 가격통계 병합 |
| `src/validation.py` (신규) | 시간 홀드아웃 백테스트 + Moran's I |
| `src/briefing.py` (신규) | 단지 컨텍스트 + 구조화 JSON 브리핑(gpt-5.4-mini) + 규칙기반 폴백 |
| `CLAUDE.md` (수정) | LLM 공급자 OpenAI 정합성 + Engineering Guidelines |
| `tests/` | 신규 단위 테스트 |

---

## Task 0: 브랜치 생성

- [ ] **Step 1:**
```bash
cd "C:/Users/ubumi/Desktop/2. 코딩연습/0. 2026/cursor/0401_01_commute"
git checkout main && git checkout -b feat/validation-and-llm
git branch --show-current
```
Expected: `feat/validation-and-llm`

---

## Task 1: `value_model.add_complex_price_stats` + run_model 배선

**Files:**
- Modify: `src/value_model.py` (함수 추가)
- Modify: `run_model.py` (호출)
- Test: `tests/test_value_model.py` (테스트 추가)

- [ ] **Step 1: Append test** to END of `tests/test_value_model.py`:
```python
from src.value_model import add_complex_price_stats


def test_add_complex_price_stats():
    scored = pd.DataFrame(
        {
            "구": ["A", "A", "A"],
            "법정동": ["x", "x", "x"],
            "아파트명": ["P", "P", "Q"],
            "price_per_sqm": [1000.0, 1200.0, 800.0],
            "거래금액": [50000, 60000, 40000],
        }
    )
    comp = pd.DataFrame({"구": ["A", "A"], "법정동": ["x", "x"], "아파트명": ["P", "Q"]})
    out = add_complex_price_stats(comp, scored)
    p = out.set_index("아파트명")
    assert p.loc["P", "avg_price_per_sqm"] == 1100.0   # (1000+1200)/2
    assert p.loc["Q", "avg_price_per_sqm"] == 800.0
    assert p.loc["P", "median_amount"] == 55000.0       # median(50000,60000)
```

- [ ] **Step 2: Run** `py -m pytest tests/test_value_model.py::test_add_complex_price_stats -v` → FAIL (ImportError).

- [ ] **Step 3: Append function** to END of `src/value_model.py`:
```python
def add_complex_price_stats(comp: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """단지별 실제 평당가·중위 거래금액을 comp에 병합 (LLM·대시보드용)."""
    stats = (
        scored.groupby(["구", "법정동", "아파트명"])
        .agg(
            avg_price_per_sqm=("price_per_sqm", "mean"),
            median_amount=("거래금액", "median"),
        )
        .reset_index()
    )
    return comp.merge(stats, on=["구", "법정동", "아파트명"], how="left")
```

- [ ] **Step 4: Wire into `run_model.py`.** Find the line:
```python
    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    print(f"단지 수: {len(comp):,}개")
```
Add the price-stats import at the top with the other value_model imports — change:
```python
from src.value_model import fit_quality_model, aggregate_to_complex, classify_zones
```
to:
```python
from src.value_model import (
    fit_quality_model,
    aggregate_to_complex,
    classify_zones,
    add_complex_price_stats,
)
```
And right AFTER the `print(f"단지 수: ...")` line, add:
```python
    comp = add_complex_price_stats(comp, scored)
```

- [ ] **Step 5: Run** `py -m pytest tests/ -q` → ALL pass. (Do NOT run full run_model here — geocoding is slow; it is regenerated in Task 6 smoke.)

- [ ] **Step 6: Commit**
```bash
git add src/value_model.py run_model.py tests/test_value_model.py
git commit -m "feat: add complex-level price statistics to model output"
```

---

## Task 2: `validation.py` — 시간 홀드아웃 백테스트

**Files:**
- Create: `src/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write test** — create `tests/test_validation.py`:
```python
import numpy as np
import pandas as pd
from src.validation import backtest


def _make_clean(months, complexes):
    rows = []
    for ym in months:
        for apt, ppsqm in complexes:
            rows.append({
                "구": "A", "법정동": "x", "아파트명": apt,
                "전용면적": 84.0, "층": 5, "age": 10,
                "area_band": "60-85", "거래년월": ym,
                "price_per_sqm": ppsqm, "거래금액": ppsqm * 84,
            })
    return pd.DataFrame(rows)


def test_backtest_splits_and_returns_zone_stats():
    months = [202509, 202510, 202511, 202512, 202601, 202602, 202603, 202604, 202605, 202606]
    clean = _make_clean(months, [("P", 1000.0), ("Q", 1500.0), ("R", 2000.0)])
    commute = pd.DataFrame({"구": ["A"], "법정동": ["x"], "commute_minutes": [30]})
    out = backtest(clean, commute, train_months=7, sigma_mult=1.0,
                   min_tx=1, recency_months=6, shrinkage_k=5)
    # 마지막 3개월(202604~202606) × 3단지 = 9건이 test
    assert out["n_test"] == 9
    assert out["n_train"] == 21
    for z in ["Blue", "Gray", "Red"]:
        assert z in out
```

- [ ] **Step 2: Run** `py -m pytest tests/test_validation.py -v` → FAIL.

- [ ] **Step 3: Implement** — create `src/validation.py`:
```python
"""저평가 신호의 자기검증: 시간 홀드아웃 백테스트 + 공간 자기상관(Moran's I)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from src.value_model import aggregate_to_complex, classify_zones, ym_subtract_months


def _unit_design(df: pd.DataFrame) -> pd.DataFrame:
    """시점 더미 없는 단위특성 설계행렬(백테스트용 — 미래 기간 예측 가능)."""
    X = pd.concat(
        [
            np.log(df["전용면적"]).rename("log_area"),
            df["층"].astype(float).rename("floor"),
            df["age"].astype(float).rename("age"),
            (df["age"].astype(float) ** 2).rename("age2"),
            pd.get_dummies(df["area_band"], prefix="band", drop_first=True),
        ],
        axis=1,
    )
    return X.astype(float)


def backtest(clean: pd.DataFrame, commute: pd.DataFrame, *, train_months,
             sigma_mult, min_tx, recency_months, shrinkage_k) -> dict:
    """앞 기간으로 zone 산출 → 뒤 3개월 실현가로 신호 유효성 검증.

    Blue(저평가 후보)의 이후 실현 잔차가 Gray/Red보다 높으면 신호가 유효.
    """
    latest = int(clean["거래년월"].max())
    test_cut = ym_subtract_months(latest, 3)  # 이보다 큰 거래년월 = test
    train = clean[clean["거래년월"] <= test_cut].copy()
    test = clean[clean["거래년월"] > test_cut].copy()
    if train.empty or test.empty:
        return {"error": "insufficient split", "n_train": len(train), "n_test": len(test)}

    Xtr = _unit_design(train)
    ytr = np.log(train["price_per_sqm"])
    model = LinearRegression().fit(Xtr, ytr)
    train = train.copy()
    train["resid"] = ytr - model.predict(Xtr)

    comp = aggregate_to_complex(train, k=shrinkage_k)
    zones = classify_zones(comp, commute, sigma_mult, min_tx, recency_months,
                           int(train["거래년월"].max()))
    zone_map = zones.set_index(["구", "법정동", "아파트명"])["zone"].to_dict()

    Xte = _unit_design(test).reindex(columns=Xtr.columns, fill_value=0)
    test = test.copy()
    test["realized_resid"] = np.log(test["price_per_sqm"]) - model.predict(Xte)
    test["zone"] = [
        zone_map.get((g, d, a), "Unknown")
        for g, d, a in zip(test["구"], test["법정동"], test["아파트명"])
    ]

    out = {"n_train": len(train), "n_test": len(test)}
    for z in ["Blue", "Gray", "Red"]:
        sub = test[test["zone"] == z]
        out[z] = {
            "n": int(len(sub)),
            "mean_realized_resid": float(sub["realized_resid"].mean()) if len(sub) else None,
        }
    b = out["Blue"]["mean_realized_resid"] or 0.0
    g = out["Gray"]["mean_realized_resid"] or 0.0
    out["blue_minus_gray"] = b - g
    out["signal_valid"] = b > g
    return out
```

- [ ] **Step 4: Run** `py -m pytest tests/test_validation.py -v` → PASS. Debug until green.

- [ ] **Step 5: Commit**
```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: add temporal-holdout backtest for undervaluation signal"
```

---

## Task 3: `validation.py` — Moran's I (수동 구현)

**Files:**
- Modify: `src/validation.py` (함수 추가)
- Test: `tests/test_validation.py` (테스트 추가)

- [ ] **Step 1: Append test** to END of `tests/test_validation.py`:
```python
from src.validation import morans_i


def test_morans_i_detects_clustering():
    # 좌표를 한 줄로 놓고 값을 좌표 순서대로 증가 → 강한 양의 공간 자기상관
    n = 60
    coords = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    clustered = np.linspace(0, 1, n)
    I_clustered, p_clustered = morans_i(clustered, coords, k=4, n_perm=199, seed=0)
    assert I_clustered > 0.5
    assert p_clustered < 0.05


def test_morans_i_random_near_zero():
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.random((n, 2))
    vals = rng.random(n)
    I_rand, _ = morans_i(vals, coords, k=4, n_perm=199, seed=1)
    assert abs(I_rand) < 0.3
```

- [ ] **Step 2: Run** `py -m pytest tests/test_validation.py -k morans -v` → FAIL (ImportError).

- [ ] **Step 3: Append functions** to END of `src/validation.py`:
```python
def _knn_neighbors(coords: np.ndarray, k: int) -> np.ndarray:
    """각 점의 k 최근접 이웃 인덱스 (자기 자신 제외)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]  # 0번째는 자기 자신


def _morans_i_stat(values: np.ndarray, neighbors: np.ndarray) -> float:
    """행 표준화 KNN 가중치 기반 Moran's I 통계량."""
    z = values - values.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return 0.0
    num = np.sum(z * z[neighbors].mean(axis=1))
    return float(num / denom)


def morans_i(values, coords, k=8, n_perm=199, seed=0):
    """Moran's I와 순열 기반 p-value 반환.

    values: 1D array, coords: (n,2) 좌표. KNN(k) 행표준화 가중치.
    유의한 양의 I → 인접 지역 잔차가 서로 상관(군집 아티팩트 가능성).
    """
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    neighbors = _knn_neighbors(coords, k)
    observed = _morans_i_stat(values, neighbors)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(values)
        if _morans_i_stat(perm, neighbors) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return observed, p_value
```

- [ ] **Step 4: Run** `py -m pytest tests/test_validation.py -v` → ALL pass. Debug until green.

- [ ] **Step 5: Commit**
```bash
git add src/validation.py tests/test_validation.py
git commit -m "feat: add Moran's I spatial autocorrelation (manual numpy/knn impl)"
```

---

## Task 4: `briefing.py` — 구조화 JSON 브리핑 (gpt-5.4-mini)

**Files:**
- Create: `src/briefing.py`
- Test: `tests/test_briefing.py`

- [ ] **Step 1: Write test** — create `tests/test_briefing.py`:
```python
import json
import pandas as pd
from src.briefing import build_context, get_briefing, rule_based_briefing


def _zones():
    return pd.DataFrame(
        {
            "구": ["노원구", "도봉구"],
            "법정동": ["상계동", "창동"],
            "아파트명": ["A아파트", "B아파트"],
            "zone": ["Blue", "Blue"],
            "commute_minutes": [45, 50],
            "avg_price_per_sqm": [1500.0, 1400.0],
            "final_resid": [-0.3, -0.25],
            "deviation_pct": [-30.0, -25.0],
            "n": [12, 8],
            "confidence": ["높음", "보통"],
            "subway_dist_km": [0.3, 0.6],
        }
    )


def _prefs():
    return {"budget_type": "전세", "budget_amount": 50000,
            "max_commute": 60, "priorities": ["통근 우선"]}


def test_build_context_filters_blue_and_lists_candidates():
    ctx, cand = build_context(_zones(), _prefs())
    assert "A아파트" in ctx
    assert len(cand) == 2
    assert (cand["zone"] == "Blue").all()


def test_get_briefing_parses_structured_json():
    payload = {
        "recommendations": [
            {"rank": 1, "구": "노원구", "법정동": "상계동", "아파트명": "A아파트",
             "reason": {"가격": "x", "통근": "y", "거래신뢰도": "z", "주의": "w"},
             "avg_price_per_sqm": 1500.0, "commute_min": 45, "residual": -0.3,
             "risk_notes": ["거래 적음"]}
        ],
        "disclaimer": "참고용",
    }
    fake_caller = lambda model, system, user: json.dumps(payload)
    out = get_briefing("ctx", _prefs(), "KEY", caller=fake_caller)
    assert out["recommendations"][0]["아파트명"] == "A아파트"
    assert out["disclaimer"] == "참고용"


def test_get_briefing_returns_none_on_error():
    def boom(model, system, user):
        raise RuntimeError("api down")
    out = get_briefing("ctx", _prefs(), "KEY", caller=boom)
    assert out is None


def test_rule_based_briefing_matches_schema_shape():
    _, cand = build_context(_zones(), _prefs())
    out = rule_based_briefing(cand, _prefs())
    assert "recommendations" in out and "disclaimer" in out
    assert len(out["recommendations"]) >= 1
    rec = out["recommendations"][0]
    assert set(["rank", "구", "법정동", "아파트명", "reason", "risk_notes"]).issubset(rec)
```

- [ ] **Step 2: Run** `py -m pytest tests/test_briefing.py -v` → FAIL.

- [ ] **Step 3: Implement** — create `src/briefing.py`:
```python
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
```

- [ ] **Step 4: Run** `py -m pytest tests/test_briefing.py -v` → 5 PASS. Debug until green.

- [ ] **Step 5: Commit**
```bash
git add src/briefing.py tests/test_briefing.py
git commit -m "feat: add structured-JSON LLM briefing with gpt-5.4-mini and rule fallback"
```

---

## Task 5: `CLAUDE.md` — LLM 공급자 정합성 + Engineering Guidelines

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1:** In `CLAUDE.md`, under "## Tech Stack", change the LLM line from:
```
- **LLM:** Anthropic Claude API
```
to:
```
- **LLM:** OpenAI API (gpt-5.4-mini, 구조화 JSON 출력)
```

- [ ] **Step 2:** Under "## Environment Variables", change `ANTHROPIC_API_KEY` line:
```
- `ANTHROPIC_API_KEY` — Anthropic API 키
```
to:
```
- `OPENAI_API_KEY` — OpenAI API 키 (LLM 브리핑)
```

- [ ] **Step 3:** Under "## Architecture", in the Phase 4 (LLM 브리핑) description, change `OpenAI API(GPT-4o)에 주입` to `OpenAI API(gpt-5.4-mini)에 주입`. And in the "## Key External APIs" section, change the `Anthropic Claude` bullet to `OpenAI gpt-5.4-mini: 구조화 JSON 출력으로 할루시네이션 방지, 데이터 근거 제약 필수`.

- [ ] **Step 4:** Append a new section at the END of `CLAUDE.md`:
```markdown
## Additional Engineering Guidelines

### Data Quality
- 원본 API 응답(`data/raw/`)과 정제 데이터를 분리하고, 분석 전 스키마를 검증한다.
- 거래건수가 부족한 단지는 저평가로 단정하지 않는다(Blue 신뢰 게이트).

### Commute API
- 모든 카카오 호출은 캐시하고, 동일 origin-dest는 재호출하지 않는다.
- 재시도(backoff)·타임아웃·일일 호출 가드를 둔다.

### Modeling
- 베이스라인: 통근→평당가. 강화: 2단계 헤도닉(면적·평형·층·연식·시점 통제 → 통근+지하철 대비 잔차).
- Blue 분류는 잔차 + 거래건수 + 최근성을 함께 본다.
- 표본이 부족하면 저평가로 단정하지 않는다.

### Validation
- 시간 홀드아웃 백테스트로 신호 유효성을 점검하고, Moran's I로 공간 자기상관(군집 아티팩트)을 검정한다.

### LLM Briefing
- 제공된 단지·분석 데이터만 사용. 단지명·가격·통근·지역을 창작 금지.
- 추천마다 risk_notes 포함. 사용자 표시 전 구조화 JSON으로 받는다.
```

- [ ] **Step 5: Verify** the file is valid markdown (skim it). Commit:
```bash
git add CLAUDE.md
git commit -m "docs: reconcile LLM provider to OpenAI and add engineering guidelines"
```

---

## Task 6: 실데이터 스모크 (백테스트 + Moran's I + 실 LLM 호출)

**Files:**
- Create: `validate_and_brief.py` (스모크 스크립트)

- [ ] **Step 1: 먼저 가격통계 포함해 산출물 재생성.** Run `py run_model.py`.
Expected: 정상 완료, `data/complex_zones.csv`에 `avg_price_per_sqm, median_amount` 컬럼이 생김. (지오코딩은 캐시 사용 + 다음 배치 일부; 수 분 소요 가능, 완료까지 대기.)
Verify:
```bash
py -c "import pandas as pd; d=pd.read_csv('data/complex_zones.csv'); print('avg_price_per_sqm' in d.columns, 'median_amount' in d.columns)"
```
Expected: `True True`

- [ ] **Step 2: Create `validate_and_brief.py`:**
```python
"""검증·브리핑 스모크: 백테스트 + Moran's I + LLM 브리핑."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import (ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
                    RECENCY_MONTHS, SHRINKAGE_K)
from src.preprocessor import preprocess
from src.commute import load_dong_commute
from src.validation import backtest, morans_i
from src.briefing import build_context, get_briefing, rule_based_briefing


def main():
    load_dotenv(ROOT / ".env")
    data_dir = ROOT / "data"
    clean = preprocess(pd.read_csv(data_dir / "seoul_apt_transactions.csv"))
    commute = load_dong_commute(data_dir / "commute_cache.json")

    print("=" * 50)
    print("백테스트 (앞기간 학습 → 뒤 3개월 검증)")
    bt = backtest(clean, commute, train_months=9, sigma_mult=ZONE_SIGMA,
                  min_tx=MIN_TRANSACTIONS_PER_COMPLEX, recency_months=RECENCY_MONTHS,
                  shrinkage_k=SHRINKAGE_K)
    print(bt)

    print("=" * 50)
    print("Moran's I (최종 잔차 공간 자기상관)")
    zones = pd.read_csv(data_dir / "complex_zones.csv").dropna(subset=["lat", "lon", "final_resid"])
    zones = zones.drop_duplicates(subset=["lat", "lon"])  # 동일 좌표(동 폴백) 중복 제거
    I, p = morans_i(zones["final_resid"].values,
                    zones[["lat", "lon"]].values, k=8, n_perm=199, seed=0)
    print(f"Moran's I = {I:.4f}, p = {p:.4f} "
          f"({'유의한 공간 자기상관' if p < 0.05 else '약함'})")

    print("=" * 50)
    print("LLM 브리핑 (gpt-5.4-mini)")
    zones_full = pd.read_csv(data_dir / "complex_zones.csv")
    prefs = {"budget_type": "전세", "budget_amount": 50000,
             "max_commute": 50, "priorities": ["통근 우선"]}
    ctx, cand = build_context(zones_full, prefs)
    api_key = os.getenv("OPENAI_API_KEY")
    result = get_briefing(ctx, prefs, api_key) if api_key else None
    if result is None:
        print("→ 규칙기반 폴백")
        result = rule_based_briefing(cand, prefs)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run** `py validate_and_brief.py`. Report FULL output. Expected:
  - 백테스트: dict with n_train/n_test, Blue/Gray/Red mean_realized_resid, blue_minus_gray, signal_valid. **Report whether signal_valid is True** (Blue realized higher than Gray) — if False, that's an honest finding to report, not a failure.
  - Moran's I: a value and p. **Report I and p** (positive significant I means spatial clustering exists — expected for real estate; it's a caveat to surface, not a crash).
  - LLM: either a structured JSON briefing (real gpt-5.4-mini call succeeded) OR "규칙기반 폴백" with the loud WARN line (if the OpenAI key/model failed). Report which path and the first part of the JSON.

- [ ] **Step 4: Commit**
```bash
git add validate_and_brief.py
git commit -m "feat: add validation+briefing smoke script"
```

---

## Self-Review (작성자 점검)

**스펙 커버리지 (Plan 3 = 구현순서 ⑥⑦ + 데이터품질 일부):**
- §5.4 단지 가격통계 → Task 1 ✓
- §7 백테스트 → Task 2 ✓
- §5.4/§7 Moran's I 공간 자기상관 (PySAL 대신 numpy/KNN 직접 구현 — 3.14 휠 리스크 회피) → Task 3 ✓
- §5.5 LLM gpt-5.4-mini + 구조화 JSON + 프롬프트 규칙 + 폴백 가시화 → Task 4 ✓
- §5.5 CLAUDE.md 정합성 + §8.2 Engineering Guidelines → Task 5 ✓
- 실데이터 검증 → Task 6 ✓

**Plan 3 범위 밖(Plan 4):** 대시보드 정직성 레이어·랭킹·링크아웃·리스크 플래그 UI, 교통호재(transit_projects.json) + 오버레이, map_builder 단지 마커, 기존 `llm_briefing.py`→`briefing.py` 전환 및 app.py 배선.

**타입/하위호환:** `aggregate_to_complex`는 변경하지 않음(기존 테스트 보존). `add_complex_price_stats`는 별도 함수. `classify_zones`는 백테스트에서 그대로 재사용. Moran's I·backtest는 numpy/sklearn만 사용(설치 확인됨). `briefing.get_briefing`은 caller 주입으로 네트워크 없이 테스트.

**플레이스홀더:** 없음.

---

## Execution Handoff
Plan 3 실행 후 Plan 4(대시보드·정직성 레이어·교통호재)를 작성한다.
