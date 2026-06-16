"""Commute-Value Finder — 메인 실행 파이프라인"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from config import COMPANY_NAME


def _check_file(path: Path, label: str) -> bool:
    """파일 존재 여부 확인, 없으면 에러 출력"""
    if path.exists():
        return True
    print(f"[ERROR] {label} 파일 없음: {path}")
    return False


def run_phase0():
    """Phase 0: 국토부 실거래가 데이터 수집"""
    print("\n" + "=" * 55)
    print("Phase 0: 국토부 실거래가 데이터 수집")
    print("=" * 55)
    from src.data_collector import collect_all
    df = collect_all()
    if df.empty:
        print("[ERROR] 데이터 수집 실패. API 키와 네트워크를 확인하세요.")
        return False
    print(f"수집 완료: {len(df):,}건")
    return True


def run_phase1():
    """Phase 1: 좌표 변환 + 히트맵 지도"""
    print("\n" + "=" * 55)
    print("Phase 1: 지오코딩 + Folium 지도 생성")
    print("=" * 55)
    root = Path(__file__).resolve().parent
    if not _check_file(root / "data" / "seoul_apt_transactions.csv", "실거래가"):
        print("  → Phase 0을 먼저 실행하세요.")
        return False
    from src.map_builder import main as mb_main
    mb_main()
    return True


def run_phase2():
    """Phase 2: 통근 시간 + 잔차 분석 + Zone 지도"""
    print("\n" + "=" * 55)
    print("Phase 2: 통근 시간 수집 + Zone 분류")
    print("=" * 55)
    root = Path(__file__).resolve().parent
    if not _check_file(root / "data" / "seoul_apt_geocoded.csv", "좌표 데이터"):
        print("  → Phase 1을 먼저 실행하세요.")
        return False
    from src.zone_analyzer import main as za_main
    za_main()
    return True


def run_phase3(demo: bool = False):
    """Phase 3: LLM 브리핑"""
    print("\n" + "=" * 55)
    print("Phase 3: AI 브리핑 생성")
    print("=" * 55)
    root = Path(__file__).resolve().parent
    if not _check_file(root / "data" / "dong_with_commute.csv", "통근 데이터"):
        print("  → Phase 2를 먼저 실행하세요.")
        return False
    from src.llm_briefing import main as lb_main
    lb_main(demo=demo)
    return True


def main():
    """메인 메뉴"""
    print("=== Commute-Value Finder ===")
    print(f"회사 위치: {COMPANY_NAME} (config.py에서 변경 가능)\n")
    print("1. 전체 분석 실행 (Phase 0~3 순서대로)")
    print("2. 지도만 업데이트 (데이터 수집 건너뜀)")
    print("3. LLM 브리핑만 실행")

    choice = input("\n선택: ").strip()

    if choice == "1":
        if not run_phase0(): return
        if not run_phase1(): return
        if not run_phase2(): return
        run_phase3()
    elif choice == "2":
        if not run_phase1(): return
        if not run_phase2(): return
        run_phase3()
    elif choice == "3":
        run_phase3()
    else:
        print("1, 2, 3 중 하나를 선택하세요.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_phase3(demo=True)
    else:
        main()
