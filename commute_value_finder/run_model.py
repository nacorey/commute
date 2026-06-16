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

    raw = pd.read_csv(src_csv, encoding="utf-8-sig")
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
