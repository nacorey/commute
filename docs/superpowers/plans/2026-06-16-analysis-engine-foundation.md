# Analysis Engine Foundation Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 단변량(통근→평당가) 분류를, 거래 건별 특성을 통제한 2단계 헤도닉 모델로 교체해 "통계적으로 방어 가능한 저평가 단지 후보"를 산출한다.

**Architecture:** 정제(`preprocessor.py`) → 1단계 헤도닉(단위특성·평형·시점 통제, 거래 잔차) → 단지 집계 + 축소추정(입지가치지수) → 2단계(통근 대비 저평가, Blue 신뢰 게이트) → `data/complex_zones.csv`. 이 Plan은 **기존 동 단위 통근 캐시**를 사용하며, 지하철 접근성·단지 지오코딩은 Plan 2에서 결합한다.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn (LinearRegression), pytest. (신규 무거운 의존성 없음 — Python 3.14에서 검증된 스택만 사용. statsmodels/PySAL은 Plan 3에서 호환성 확인 후 도입.)

---

## File Structure

| 파일 | 책임 |
|---|---|
| `commute_value_finder/src/preprocessor.py` (신규) | 실거래 CSV 정제·검증·중복제거·이상치·파생컬럼 |
| `commute_value_finder/src/value_model.py` (신규) | 2단계 헤도닉 모델, 단지 집계, zone 분류 |
| `commute_value_finder/src/data_collector.py` (수정) | 해제여부(취소거래) 필드 수집 |
| `commute_value_finder/config.py` (수정) | 신규 파라미터 |
| `commute_value_finder/run_model.py` (신규) | 엔드투엔드 오케스트레이터(CLI) |
| `commute_value_finder/tests/` (신규) | pytest 단위 테스트 |

**데이터 컬럼(원본 `seoul_apt_transactions.csv`):** `지역코드, 구, 아파트명, 법정동, 전용면적, 거래금액, 층, 건축년도, 년, 월, price_per_sqm, 거래년월`. 정제 후 파생: `age, area_band`. 수집기 수정 후: `해제여부, 해제사유발생일`.

---

## Task 0: 프로젝트를 git으로 초기화 + 테스트 셋업

**Files:**
- Create: `commute_value_finder/tests/__init__.py`
- Create: `commute_value_finder/tests/conftest.py`

- [ ] **Step 1: git 저장소 초기화**

Run:
```bash
cd "C:/Users/ubumi/Desktop/2. 코딩연습/0. 2026/cursor/0401_01_commute"
git init
git add -A
git commit -m "chore: snapshot existing project before analysis engine upgrade"
```
Expected: 초기 커밋 생성. (이미 git이면 이 Task는 건너뛴다.)

- [ ] **Step 2: pytest 설치 확인**

Run: `cd commute_value_finder && python -m pip install pytest`
Expected: `Successfully installed pytest` 또는 already satisfied.

- [ ] **Step 3: tests 패키지 생성**

Create `commute_value_finder/tests/__init__.py` (빈 파일).

Create `commute_value_finder/tests/conftest.py`:
```python
"""pytest 공통 픽스처."""
import sys
from pathlib import Path

# src 임포트를 위해 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


@pytest.fixture
def sample_transactions():
    """정제·모델 테스트용 소형 거래 데이터프레임."""
    return pd.DataFrame(
        {
            "구": ["강남구", "강남구", "노원구", "노원구", "노원구"],
            "법정동": ["삼성동", "삼성동", "상계동", "상계동", "상계동"],
            "아파트명": ["A아파트", "A아파트", "B아파트", "B아파트", "C아파트"],
            "전용면적": [84.0, 59.0, 84.0, 84.0, 49.0],
            "거래금액": [200000, 150000, 60000, 62000, 40000],
            "층": [10, 3, 5, 15, 2],
            "건축년도": [2010, 2010, 1995, 1995, 1988],
            "년": [2026, 2026, 2026, 2025, 2025],
            "월": [3, 1, 5, 12, 6],
        }
    )
```

- [ ] **Step 4: 셋업 커밋**

```bash
cd commute_value_finder
git add tests/__init__.py tests/conftest.py
git commit -m "test: add pytest scaffolding and sample fixture"
```

---

## Task 1: `preprocessor.py` — 정제·파생 컬럼

**Files:**
- Create: `commute_value_finder/src/preprocessor.py`
- Test: `commute_value_finder/tests/test_preprocessor.py`

- [ ] **Step 1: 실패하는 테스트 작성**

Create `commute_value_finder/tests/test_preprocessor.py`:
```python
import numpy as np
import pandas as pd
from src.preprocessor import (
    coerce_amount,
    add_derived_columns,
    remove_invalid,
    remove_duplicates,
    winsorize_price,
    preprocess,
)


def test_coerce_amount_handles_comma_string():
    assert coerce_amount("84,500") == 84500
    assert coerce_amount(84500) == 84500
    assert np.isnan(coerce_amount("abc"))


def test_add_derived_columns(sample_transactions):
    out = add_derived_columns(sample_transactions)
    # price_per_sqm = 거래금액 / 전용면적
    assert out.loc[0, "price_per_sqm"] == 200000 / 84.0
    # age = 년 - 건축년도
    assert out.loc[0, "age"] == 2026 - 2010
    # 거래년월 = 년*100 + 월 (정수)
    assert out.loc[0, "거래년월"] == 202603
    # area_band: 84 -> "60-85", 59 -> "~60", 49 -> "~60"
    assert out.loc[0, "area_band"] == "60-85"
    assert out.loc[1, "area_band"] == "~60"


def test_remove_invalid_drops_nonpositive_area_and_negative_age():
    df = pd.DataFrame(
        {
            "전용면적": [84.0, 0.0, 84.0],
            "price_per_sqm": [100.0, 100.0, 100.0],
            "age": [5, 5, -3],
        }
    )
    out = remove_invalid(df)
    assert len(out) == 1


def test_remove_duplicates():
    df = pd.DataFrame(
        {
            "아파트명": ["A", "A"],
            "법정동": ["삼성동", "삼성동"],
            "층": [10, 10],
            "전용면적": [84.0, 84.0],
            "거래년월": [202603, 202603],
            "거래금액": [200000, 200000],
        }
    )
    assert len(remove_duplicates(df)) == 1


def test_winsorize_price_clips_extremes():
    df = pd.DataFrame({"price_per_sqm": list(range(1, 101))})
    out = winsorize_price(df, (1, 99))
    # 1퍼센타일/99퍼센타일 밖은 잘려야 함 (최솟값/최댓값이 원본보다 안쪽)
    assert out["price_per_sqm"].min() >= np.percentile(range(1, 101), 1)
    assert out["price_per_sqm"].max() <= np.percentile(range(1, 101), 99)


def test_preprocess_end_to_end(sample_transactions):
    out = preprocess(sample_transactions)
    assert "price_per_sqm" in out.columns
    assert "age" in out.columns
    assert "area_band" in out.columns
    assert (out["전용면적"] > 0).all()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_preprocessor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.preprocessor'`

- [ ] **Step 3: `preprocessor.py` 구현**

Create `commute_value_finder/src/preprocessor.py`:
```python
"""실거래 데이터 정제·검증 레이어.

수집(data_collector)과 모델(value_model) 사이에서 정제를 전담한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import AREA_BANDS, PRICE_OUTLIER_PCT


def coerce_amount(value):
    """'84,500' 같은 문자열/숫자를 정수로. 실패 시 NaN."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, AttributeError):
        return np.nan


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """price_per_sqm, age, 거래년월(int), area_band 파생."""
    d = df.copy()
    d["거래금액"] = d["거래금액"].map(coerce_amount)
    d["전용면적"] = pd.to_numeric(d["전용면적"], errors="coerce")
    d["price_per_sqm"] = d["거래금액"] / d["전용면적"]
    d["age"] = pd.to_numeric(d["년"], errors="coerce") - pd.to_numeric(
        d["건축년도"], errors="coerce"
    )
    d["거래년월"] = (
        pd.to_numeric(d["년"], errors="coerce") * 100
        + pd.to_numeric(d["월"], errors="coerce")
    ).astype("Int64")

    # 평형 구간: AREA_BANDS = [60, 85, 135]
    bins = [-np.inf, *AREA_BANDS, np.inf]
    labels = ["~60", "60-85", "85-135", "135~"]
    d["area_band"] = pd.cut(d["전용면적"], bins=bins, labels=labels).astype(str)
    return d


def remove_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """비정상 면적(≤0)·연령(<0)·결측 평당가 제거."""
    d = df.copy()
    mask = (d["전용면적"] > 0) & (d["age"] >= 0) & d["price_per_sqm"].notna()
    return d[mask].reset_index(drop=True)


def remove_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """해제여부=='O'(계약 취소) 거래 제거. 컬럼 없으면 통과."""
    if "해제여부" not in df.columns:
        return df
    d = df.copy()
    keep = d["해제여부"].fillna("").astype(str).str.strip().str.upper() != "O"
    return d[keep].reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """동일 (아파트·법정동·층·면적·거래년월) 중복 제거."""
    keys = [c for c in ["아파트명", "법정동", "층", "전용면적", "거래년월"]
            if c in df.columns]
    return df.drop_duplicates(subset=keys).reset_index(drop=True)


def winsorize_price(df: pd.DataFrame, pct: tuple) -> pd.DataFrame:
    """price_per_sqm를 (low, high) 백분위로 절단."""
    d = df.copy()
    lo, hi = np.percentile(d["price_per_sqm"].dropna(), pct)
    d = d[(d["price_per_sqm"] >= lo) & (d["price_per_sqm"] <= hi)]
    return d.reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """전체 정제 파이프라인."""
    d = add_derived_columns(df)
    d = remove_cancelled(d)
    d = remove_invalid(d)
    d = remove_duplicates(d)
    d = winsorize_price(d, PRICE_OUTLIER_PCT)
    return d
```

- [ ] **Step 4: config에 필요한 파라미터 임시 추가 (Task 6에서 정리)**

`commute_value_finder/config.py` 끝에 추가 (Task 6에서 다른 값과 함께 정리):
```python
# ── 모델 정제 파라미터 ──
AREA_BANDS = [60, 85, 135]            # 전용면적 평형 구간 경계 (㎡)
PRICE_OUTLIER_PCT = (1, 99)           # 평당가 이상치 절단 백분위
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_preprocessor.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: 커밋**

```bash
cd commute_value_finder
git add src/preprocessor.py tests/test_preprocessor.py config.py
git commit -m "feat: add data preprocessing layer with cleaning and derived columns"
```

---

## Task 2: `data_collector.py` — 해제여부(취소거래) 필드 수집

**Files:**
- Modify: `commute_value_finder/src/data_collector.py` (FIELD_MAP, col_order)

- [ ] **Step 1: FIELD_MAP에 취소거래 필드 추가**

`commute_value_finder/src/data_collector.py`의 `FIELD_MAP`(약 31~41행)에 두 줄 추가:
```python
FIELD_MAP = {
    "sggCd": "지역코드",
    "aptNm": "아파트명",
    "umdNm": "법정동",
    "excluUseAr": "전용면적",
    "dealAmount": "거래금액",
    "floor": "층",
    "buildYear": "건축년도",
    "dealYear": "년",
    "dealMonth": "월",
    "cdealType": "해제여부",       # 'O' = 계약 취소
    "cdealDay": "해제사유발생일",
}
```

- [ ] **Step 2: col_order에 추가**

같은 파일 `col_order`(약 157~160행)를 수정:
```python
    col_order = [
        "지역코드", "구", "아파트명", "법정동", "전용면적",
        "거래금액", "층", "건축년도", "년", "월",
        "해제여부", "해제사유발생일",
    ]
```

- [ ] **Step 3: 변경이 기존 파싱을 깨지 않는지 스모크 확인**

Run: `cd commute_value_finder && python -c "from src.data_collector import FIELD_MAP; print('해제여부' in FIELD_MAP.values())"`
Expected: `True`

> 참고: 실제 재수집(`python src/data_collector.py`)은 `MOLIT_API_KEY` 필요. 기존 CSV에 컬럼이 없어도 `preprocessor.remove_cancelled`가 통과하도록 설계됨(하위 호환).

- [ ] **Step 4: 커밋**

```bash
cd commute_value_finder
git add src/data_collector.py
git commit -m "feat: collect contract-cancellation fields from MOLIT API"
```

---

## Task 3: `value_model.py` — 1단계 헤도닉(거래 잔차)

**Files:**
- Create: `commute_value_finder/src/value_model.py`
- Test: `commute_value_finder/tests/test_value_model.py`

- [ ] **Step 1: 실패하는 테스트 작성**

Create `commute_value_finder/tests/test_value_model.py`:
```python
import numpy as np
import pandas as pd
from src.preprocessor import preprocess
from src.value_model import fit_quality_model


def test_fit_quality_model_returns_residuals(sample_transactions):
    clean = preprocess(sample_transactions)
    model, scored, r2 = fit_quality_model(clean)
    # 잔차 컬럼이 생기고, 행 수가 유지된다
    assert "resid" in scored.columns
    assert len(scored) == len(clean)
    # 잔차 평균은 OLS 특성상 0에 가깝다
    assert abs(scored["resid"].mean()) < 1e-6
    # R²는 0~1 범위
    assert 0.0 <= r2 <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_value_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.value_model'`

- [ ] **Step 3: `fit_quality_model` 구현**

Create `commute_value_finder/src/value_model.py`:
```python
"""2단계 헤도닉 가치 모델.

1단계: 거래 건별 단위특성·평형·시점을 통제해 가격 잔차를 얻는다.
2단계: 단지 입지가치지수를 통근시간으로 회귀해 "저평가" 잔차를 얻는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """헤도닉 설계행렬: log(면적)+층+age+age²+평형더미+거래년월더미."""
    X = pd.concat(
        [
            np.log(df["전용면적"]).rename("log_area"),
            df["층"].astype(float).rename("floor"),
            df["age"].astype(float).rename("age"),
            (df["age"].astype(float) ** 2).rename("age2"),
            pd.get_dummies(df["area_band"], prefix="band", drop_first=True),
            pd.get_dummies(df["거래년월"].astype(str), prefix="ym", drop_first=True),
        ],
        axis=1,
    )
    return X.astype(float)


def fit_quality_model(df: pd.DataFrame):
    """1단계 헤도닉. 반환: (model, df+resid, r2).

    위치(구/동/단지) 변수는 넣지 않는다 — 위치 편차가 우리가 찾는 신호이므로.
    """
    d = df.copy()
    y = np.log(d["price_per_sqm"])
    X = _design_matrix(d)
    model = LinearRegression().fit(X, y)
    d["resid"] = y - model.predict(X)
    r2 = float(model.score(X, y))
    return model, d, r2
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_value_model.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 커밋**

```bash
cd commute_value_finder
git add src/value_model.py tests/test_value_model.py
git commit -m "feat: add stage-1 hedonic quality model"
```

---

## Task 4: `value_model.py` — 단지 집계 + 축소추정

**Files:**
- Modify: `commute_value_finder/src/value_model.py` (함수 추가)
- Test: `commute_value_finder/tests/test_value_model.py` (테스트 추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`commute_value_finder/tests/test_value_model.py` 끝에 추가:
```python
from src.value_model import aggregate_to_complex


def test_aggregate_to_complex_shrinks_small_samples():
    # 단지 X: 거래 1건, 잔차 +1.0 (동 평균은 0 근처로 끌려가야 함)
    scored = pd.DataFrame(
        {
            "구": ["강남구", "강남구", "강남구"],
            "법정동": ["삼성동", "삼성동", "삼성동"],
            "아파트명": ["X", "Y", "Y"],
            "거래년월": [202601, 202602, 202603],
            "resid": [1.0, 0.0, 0.0],
        }
    )
    comp = aggregate_to_complex(scored, k=5)
    x_row = comp[comp["아파트명"] == "X"].iloc[0]
    # n=1, r_c=1.0, r_dong=(1+0+0)/3≈0.333, k=5
    # 입지가치지수 = (1*1.0 + 5*0.333)/(1+5) ≈ 0.444  (1.0보다 동평균 쪽으로 수축)
    assert x_row["n"] == 1
    assert x_row["입지가치지수"] < 1.0
    assert x_row["last_ym"] == 202601
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_value_model.py::test_aggregate_to_complex_shrinks_small_samples -v`
Expected: FAIL — `ImportError: cannot import name 'aggregate_to_complex'`

- [ ] **Step 3: `aggregate_to_complex` 구현**

`commute_value_finder/src/value_model.py` 끝에 추가:
```python
def aggregate_to_complex(scored: pd.DataFrame, k: float) -> pd.DataFrame:
    """거래 잔차를 단지 단위로 집계 + 동 평균으로 Empirical Bayes 수축.

    입지가치지수 = (n·r_c + k·r_dong) / (n + k)
    """
    dong = (
        scored.groupby(["구", "법정동"])["resid"].mean().rename("r_dong").reset_index()
    )
    comp = (
        scored.groupby(["구", "법정동", "아파트명"])
        .agg(
            r_c=("resid", "mean"),
            n=("resid", "size"),
            last_ym=("거래년월", "max"),
        )
        .reset_index()
    )
    comp = comp.merge(dong, on=["구", "법정동"], how="left")
    comp["입지가치지수"] = (comp["n"] * comp["r_c"] + k * comp["r_dong"]) / (
        comp["n"] + k
    )
    return comp
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_value_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
cd commute_value_finder
git add src/value_model.py tests/test_value_model.py
git commit -m "feat: aggregate residuals to complex level with empirical-bayes shrinkage"
```

---

## Task 5: `value_model.py` — 2단계 분류 + Blue 신뢰 게이트

**Files:**
- Modify: `commute_value_finder/src/value_model.py` (함수 추가)
- Test: `commute_value_finder/tests/test_value_model.py` (테스트 추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`commute_value_finder/tests/test_value_model.py` 끝에 추가:
```python
from src.value_model import ym_subtract_months, classify_zones


def test_ym_subtract_months():
    assert ym_subtract_months(202603, 6) == 202509
    assert ym_subtract_months(202601, 1) == 202512


def test_classify_zones_applies_blue_gate():
    # 두 단지 모두 강한 음의 최종잔차(저평가)지만,
    # LOWN은 거래건수 부족 → 게이트 탈락 → Gray, 후보 플래그
    comp = pd.DataFrame(
        {
            "구": ["노원구", "노원구", "강남구"],
            "법정동": ["상계동", "상계동", "삼성동"],
            "아파트명": ["GOOD", "LOWN", "PREM"],
            "입지가치지수": [-0.30, -0.30, 0.30],
            "n": [12, 2, 12],
            "last_ym": [202603, 202603, 202603],
        }
    )
    commute = pd.DataFrame(
        {
            "구": ["노원구", "강남구"],
            "법정동": ["상계동", "삼성동"],
            "commute_minutes": [50, 10],
        }
    )
    out = classify_zones(
        comp, commute, sigma_mult=1.0, min_tx=5,
        recency_months=6, latest_ym=202603,
    )
    z = out.set_index("아파트명")["zone"]
    assert z["GOOD"] == "Blue"          # 게이트 통과
    assert z["LOWN"] == "Gray"          # 거래건수 부족 → 강등
    assert out.set_index("아파트명").loc["LOWN", "blue_candidate_lowconf"]
    assert "confidence" in out.columns
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_value_model.py -k "classify or subtract" -v`
Expected: FAIL — `ImportError: cannot import name 'ym_subtract_months'`

- [ ] **Step 3: `ym_subtract_months` + `classify_zones` 구현**

`commute_value_finder/src/value_model.py` 끝에 추가:
```python
def ym_subtract_months(ym: int, months: int) -> int:
    """YYYYMM 정수에서 N개월 뺀 YYYYMM."""
    year, month = divmod(int(ym), 100)
    total = year * 12 + (month - 1) - months
    y, m = divmod(total, 12)
    return y * 100 + (m + 1)


def classify_zones(
    comp: pd.DataFrame,
    commute: pd.DataFrame,
    sigma_mult: float,
    min_tx: int,
    recency_months: int,
    latest_ym: int,
) -> pd.DataFrame:
    """2단계: 입지가치지수 ~ commute_minutes → final_resid → zone.

    Blue는 잔차 조건 + 거래건수 + 최근성 게이트를 모두 만족해야 확정.
    """
    d = comp.merge(
        commute[["구", "법정동", "commute_minutes"]], on=["구", "법정동"], how="left"
    )
    # 통근 결측은 전체 평균으로 대체(분류 안정성)
    mean_commute = d["commute_minutes"].mean()
    d["commute_minutes"] = d["commute_minutes"].fillna(mean_commute)

    model = LinearRegression().fit(d[["commute_minutes"]], d["입지가치지수"])
    d["pred_idx"] = model.predict(d[["commute_minutes"]])
    d["final_resid"] = d["입지가치지수"] - d["pred_idx"]
    d["deviation_pct"] = (d["final_resid"] * 100).round(1)  # log 잔차 ≈ % 편차

    sigma = d["final_resid"].std()
    d["zone"] = "Gray"
    d.loc[d["final_resid"] > sigma_mult * sigma, "zone"] = "Red"

    blue_resid = d["final_resid"] < -sigma_mult * sigma
    recency_cut = ym_subtract_months(latest_ym, recency_months)
    gate = blue_resid & (d["n"] >= min_tx) & (d["last_ym"] >= recency_cut)
    d.loc[gate, "zone"] = "Blue"
    d["blue_candidate_lowconf"] = blue_resid & ~gate

    d["confidence"] = pd.cut(
        d["n"], bins=[-1, 4, 9, np.inf], labels=["낮음", "보통", "높음"]
    ).astype(str)
    return d
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_value_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
cd commute_value_finder
git add src/value_model.py tests/test_value_model.py
git commit -m "feat: stage-2 zone classification with blue confidence gate"
```

---

## Task 6: `config.py` — 파라미터 정리

**Files:**
- Modify: `commute_value_finder/config.py`

- [ ] **Step 1: Task 1에서 임시로 넣은 블록을 정식 블록으로 정리**

`commute_value_finder/config.py`에서 Task 1 Step 4에 추가했던 임시 블록을 찾아 아래 정식 블록으로 대체(또는 `ZONE_SIGMA` 아래에 통합):
```python
# ── 가치 모델 파라미터 ──────────────────────────────────
AREA_BANDS = [60, 85, 135]            # 전용면적 평형 구간 경계 (㎡)
PRICE_OUTLIER_PCT = (1, 99)           # 평당가 이상치 절단 백분위
MIN_TRANSACTIONS_PER_COMPLEX = 5      # Blue 게이트: 최소 거래건수
SHRINKAGE_K = 5                       # Empirical Bayes 수축 강도
RECENCY_MONTHS = 6                    # Blue 게이트: 최근 거래 존재 기준(개월)
# ZONE_SIGMA 는 기존 정의(=1.0) 재사용
```
중복 정의가 없도록 `AREA_BANDS`/`PRICE_OUTLIER_PCT`가 한 번만 선언되게 한다.

- [ ] **Step 2: import 정상 확인**

Run: `cd commute_value_finder && python -c "import config; print(config.MIN_TRANSACTIONS_PER_COMPLEX, config.SHRINKAGE_K, config.RECENCY_MONTHS, config.ZONE_SIGMA)"`
Expected: `5 5 6 1.0`

- [ ] **Step 3: 정제·모델 테스트 재실행(파라미터 연결 확인)**

Run: `cd commute_value_finder && python -m pytest tests/ -v`
Expected: PASS (전체 통과)

- [ ] **Step 4: 커밋**

```bash
cd commute_value_finder
git add config.py
git commit -m "chore: consolidate value-model parameters in config"
```

---

## Task 7: `run_model.py` — 엔드투엔드 오케스트레이터 + 실데이터 스모크

**Files:**
- Create: `commute_value_finder/run_model.py`
- Test: `commute_value_finder/tests/test_run_model.py`

- [ ] **Step 1: 실패하는 통합 테스트 작성**

Create `commute_value_finder/tests/test_run_model.py`:
```python
import pandas as pd
from src.preprocessor import preprocess
from src.value_model import (
    fit_quality_model,
    aggregate_to_complex,
    classify_zones,
)
from config import SHRINKAGE_K, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX, RECENCY_MONTHS


def test_full_pipeline_produces_zones(sample_transactions):
    clean = preprocess(sample_transactions)
    _, scored, _ = fit_quality_model(clean)
    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    commute = pd.DataFrame(
        {
            "구": ["강남구", "노원구"],
            "법정동": ["삼성동", "상계동"],
            "commute_minutes": [10, 50],
        }
    )
    latest_ym = int(clean["거래년월"].max())
    out = classify_zones(
        comp, commute, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
        RECENCY_MONTHS, latest_ym,
    )
    assert set(out["zone"]).issubset({"Blue", "Gray", "Red"})
    assert {"입지가치지수", "final_resid", "zone", "confidence"}.issubset(out.columns)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_run_model.py -v`
Expected: FAIL (import 또는 단언 실패 — 아직 run_model 없음이 아닌 경우, 함수 조합 동작 확인용이므로 통과할 수도 있음. 통과하면 Step 3로 진행)

- [ ] **Step 3: `run_model.py` 구현**

Create `commute_value_finder/run_model.py`:
```python
"""분석 엔진 엔드투엔드 실행 (Plan 1).

기존 정제 전 거래 CSV + 기존 동 단위 통근 캐시 → 단지 단위 zone CSV.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import pandas as pd

from config import (
    SHRINKAGE_K,
    ZONE_SIGMA,
    MIN_TRANSACTIONS_PER_COMPLEX,
    RECENCY_MONTHS,
)
from src.preprocessor import preprocess
from src.value_model import fit_quality_model, aggregate_to_complex, classify_zones


def load_commute() -> pd.DataFrame:
    """기존 동 단위 통근 캐시(commute_cache.json: '구_법정동' -> 분)를 DF로."""
    path = ROOT / "data" / "commute_cache.json"
    if not path.exists():
        return pd.DataFrame(columns=["구", "법정동", "commute_minutes"])
    cache = json.loads(path.read_text("utf-8"))
    rows = []
    for key, minutes in cache.items():
        gu, dong = key.split("_", 1)
        rows.append({"구": gu, "법정동": dong, "commute_minutes": minutes})
    return pd.DataFrame(rows)


def main():
    data_dir = ROOT / "data"
    src_csv = data_dir / "seoul_apt_transactions.csv"
    if not src_csv.exists():
        print(f"[ERROR] 데이터 없음: {src_csv}")
        return

    raw = pd.read_csv(src_csv)
    print(f"원본 거래: {len(raw):,}건")

    clean = preprocess(raw)
    print(f"정제 후: {len(clean):,}건")

    model, scored, r2 = fit_quality_model(clean)
    print(f"1단계 헤도닉 R² = {r2:.4f}")

    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    print(f"단지 수: {len(comp):,}개")

    commute = load_commute()
    latest_ym = int(clean["거래년월"].max())
    zones = classify_zones(
        comp, commute, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
        RECENCY_MONTHS, latest_ym,
    )

    out_path = data_dir / "complex_zones.csv"
    zones.to_csv(out_path, index=False, encoding="utf-8-sig")
    vc = zones["zone"].value_counts()
    print(f"저장: {out_path}")
    print(f"  Blue {vc.get('Blue', 0)} / Gray {vc.get('Gray', 0)} / Red {vc.get('Red', 0)}")
    print(f"  저평가 후보(저신뢰): {int(zones['blue_candidate_lowconf'].sum())}개")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통합 테스트 통과 확인**

Run: `cd commute_value_finder && python -m pytest tests/test_run_model.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 실데이터 스모크 실행**

Run: `cd commute_value_finder && python run_model.py`
Expected: 콘솔에 `원본 거래`, `1단계 헤도닉 R²`(기존 단변량보다 높을 것), `단지 수`, `Blue/Gray/Red` 카운트가 출력되고 `data/complex_zones.csv` 생성. 에러 없이 완주.

- [ ] **Step 6: 산출물 sanity 확인**

Run: `cd commute_value_finder && python -c "import pandas as pd; d=pd.read_csv('data/complex_zones.csv'); print(d[['구','법정동','아파트명','n','입지가치지수','final_resid','zone','confidence']].sort_values('final_resid').head(10).to_string())"`
Expected: 최종잔차 최저(가장 저평가) 단지 10개가 출력. Blue로 분류된 행은 `n >= 5`여야 함(게이트 동작 확인).

- [ ] **Step 7: 커밋**

```bash
cd commute_value_finder
git add run_model.py tests/test_run_model.py
git commit -m "feat: end-to-end model orchestrator producing complex-level zones"
```

---

## Self-Review (작성자 점검 결과)

**스펙 커버리지 (Plan 1 범위 = 구현순서 ①②):**
- §5.0 취소거래 필드 수집 → Task 2 ✓
- §5.0b 정제 레이어(검증·중복·이상치·표준화 일부) → Task 1 ✓ (구/동 표준화는 데이터가 이미 한글 정규형이라 최소화; Plan 2 지오코딩 단계에서 키 매칭 시 재확인)
- §5.4 1단계 헤도닉(평형·시점 통제) → Task 3 ✓
- §5.4 단지 집계 + 축소추정 → Task 4 ✓
- §5.4 2단계 + Blue 신뢰 게이트(거래건수·최근성·confidence) → Task 5 ✓
- §5.7 파라미터 → Task 6 ✓
- 엔드투엔드 산출물(`complex_zones.csv`) → Task 7 ✓

**Plan 1 범위 밖(후속 Plan에서):** 지하철 접근성·단지 지오코딩(Plan 2), 공간자기상관 Moran's I·백테스트·LLM·대시보드·호재(Plan 3). `classify_zones`의 2단계는 Plan 2에서 `subway_dist_km`를 회귀에 추가하도록 확장 예정(현재는 commute 단독).

**타입 일관성:** `거래년월`은 전 구간 정수(YYYYMM). `resid`(거래 잔차) → `입지가치지수`(단지 수축값) → `final_resid`(2단계 잔차)로 이름이 단계별로 구분됨. `classify_zones` 시그니처가 Task 5·7에서 동일.

**플레이스홀더:** 없음 — 모든 코드 스텝에 실제 코드 포함.

---

## Execution Handoff

이 Plan 1을 실행한 뒤 Plan 2(통근·공간피처·해상도), Plan 3(검증·표현)을 이어서 작성한다.
