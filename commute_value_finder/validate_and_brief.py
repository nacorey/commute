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
