"""분석 엔진 엔드투엔드 실행 (Plan 1 + Plan 2).

거래 CSV → 정제 → 헤도닉 → 단지 집계 → 단지 지오코딩 → 지하철 접근성
→ 통근 결합 → 단지 단위 zone CSV.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

from config import (
    SHRINKAGE_K,
    ZONE_SIGMA,
    MIN_TRANSACTIONS_PER_COMPLEX,
    RECENCY_MONTHS,
    MAX_GEOCODE_CALLS,
)
from src.preprocessor import preprocess
from src.value_model import fit_quality_model, aggregate_to_complex, classify_zones
from src.commute import load_dong_commute
from src.subway_access import load_stations, add_subway_access
from src.apt_geocoder import load_dong_coords, geocode_complexes


def main():
    load_dotenv(ROOT / ".env")
    data_dir = ROOT / "data"
    src_csv = data_dir / "seoul_apt_transactions.csv"
    if not src_csv.exists():
        print(f"[ERROR] 데이터 없음: {src_csv}")
        return

    raw = pd.read_csv(src_csv)
    print(f"원본 거래: {len(raw):,}건")

    clean = preprocess(raw)
    print(f"정제 후: {len(clean):,}건")

    _, scored, r2 = fit_quality_model(clean)
    print(f"1단계 헤도닉 R² = {r2:.4f}")

    comp = aggregate_to_complex(scored, k=SHRINKAGE_K)
    print(f"단지 수: {len(comp):,}개")

    # ── 단지 지오코딩 (거래 많은 단지 우선, 캐시 + 동 좌표 폴백) ──
    comp = comp.sort_values("n", ascending=False).reset_index(drop=True)
    dong_coords = load_dong_coords(data_dir / "geocode_cache.json")
    kakao_key = os.getenv("KAKAO_API_KEY")
    if kakao_key and kakao_key.startswith("여기에"):
        kakao_key = None
    comp = geocode_complexes(comp, kakao_key, dong_coords,
                             data_dir / "apt_geocode_cache.json",
                             max_calls=MAX_GEOCODE_CALLS)
    n_coords = int(comp["lat"].notna().sum())
    print(f"좌표 확보: {n_coords:,}/{len(comp):,} 단지")

    # ── 지하철 접근성 ──
    stations = load_stations(data_dir / "subway_stations.json")
    comp = add_subway_access(comp, stations)
    n_subway = int(comp["subway_dist_km"].notna().sum())
    if n_subway:
        print(f"지하철 접근성: {n_subway:,} 단지 | "
              f"평균 최근접역 {comp['subway_dist_km'].mean():.2f}km")

    # ── 통근 결합 + 분류 ──
    commute = load_dong_commute(data_dir / "commute_cache.json")
    latest_ym = int(clean["거래년월"].max())
    zones = classify_zones(comp, commute, ZONE_SIGMA, MIN_TRANSACTIONS_PER_COMPLEX,
                           RECENCY_MONTHS, latest_ym)

    out_path = data_dir / "complex_zones.csv"
    zones.to_csv(out_path, index=False, encoding="utf-8-sig")
    vc = zones["zone"].value_counts()
    print(f"저장: {out_path}")
    print(f"  Blue {vc.get('Blue', 0)} / Gray {vc.get('Gray', 0)} / Red {vc.get('Red', 0)}")
    print(f"  저평가 후보(저신뢰): {int(zones['blue_candidate_lowconf'].sum())}개")


if __name__ == "__main__":
    main()
