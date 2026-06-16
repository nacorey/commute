"""국토교통부 아파트매매 실거래가 데이터 수집 모듈

서울시 25개 구의 최근 12개월 아파트 매매 실거래가를 수집하고
data/seoul_apt_transactions.csv로 저장한다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import SEOUL_DISTRICT_CODES, get_analysis_months
from src.utils import load_env, get_project_root


# 공공데이터포털 아파트매매 실거래 상세 자료 API
API_URL = (
    "http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/"
    "getRTMSDataSvcAptTradeDev"
)

# API 응답 필드 → 한글 컬럼 매핑
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

MAX_RETRIES = 3
ROWS_PER_PAGE = 1000
MAX_WORKERS = 5


def fetch_items(api_key: str, lawd_cd: str, deal_ymd: str) -> list[dict]:
    """한 구 + 한 달치 실거래가 수집 (페이지네이션 + 3회 재시도)"""
    all_items = []
    page_no = 1

    while True:
        resp_content = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    API_URL,
                    params={
                        "serviceKey": api_key,
                        "LAWD_CD": lawd_cd,
                        "DEAL_YMD": deal_ymd,
                        "numOfRows": ROWS_PER_PAGE,
                        "pageNo": page_no,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                resp_content = resp.content
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    return all_items
                time.sleep(attempt)

        try:
            root = ET.fromstring(resp_content)
        except ET.ParseError:
            return all_items

        result_code = root.findtext(".//resultCode")
        if result_code not in ("00", "000"):
            return all_items

        total_count = int(root.findtext(".//totalCount") or "0")
        if total_count == 0:
            return all_items

        page_items = []
        for item_elem in root.iter("item"):
            record = {}
            for child in item_elem:
                tag = child.tag
                val = (child.text or "").strip()
                # 매핑된 필드만 수집
                if tag in FIELD_MAP:
                    record[FIELD_MAP[tag]] = val
            page_items.append(record)

        all_items.extend(page_items)

        if len(all_items) >= total_count or not page_items:
            break
        page_no += 1

    return all_items


def _fetch_task(api_key: str, code: str, name: str, month: str) -> list[dict]:
    """ThreadPoolExecutor용 래퍼: 한 (구, 월) 조합 수집"""
    items = fetch_items(api_key, code, month)
    for item in items:
        item["구"] = name
    return items


def collect_all() -> pd.DataFrame:
    """서울시 25개 구 x 최근 12개월 실거래가 수집 (동시 요청)"""
    env = load_env(required_keys=["MOLIT_API_KEY"])
    api_key = env["MOLIT_API_KEY"]
    months = get_analysis_months()

    print(f"수집 기간: {months[0]} ~ {months[-1]} ({len(months)}개월)")
    print(f"대상 지역: 서울시 {len(SEOUL_DISTRICT_CODES)}개 구")
    print(f"총 요청 수: {len(SEOUL_DISTRICT_CODES) * len(months)}건 (동시 {MAX_WORKERS}개)\n")

    tasks = [
        (code, name, month)
        for code, name in SEOUL_DISTRICT_CODES.items()
        for month in months
    ]

    all_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_fetch_task, api_key, code, name, month): (code, name, month)
            for code, name, month in tasks
        }

        with tqdm(total=len(tasks), desc="수집 중", unit="건") as pbar:
            for future in as_completed(future_map):
                try:
                    items = future.result()
                    all_data.extend(items)
                except Exception:
                    pass
                pbar.update(1)

    if not all_data:
        print("[WARN] 수집된 데이터가 없습니다. API 키와 네트워크를 확인하세요.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # 컬럼 순서 정리
    col_order = [
        "지역코드", "구", "아파트명", "법정동", "전용면적",
        "거래금액", "층", "건축년도", "년", "월",
        "해제여부", "해제사유발생일",
    ]
    df = df[[c for c in col_order if c in df.columns]].copy()

    # 데이터 타입 변환
    df["거래금액"] = df["거래금액"].str.replace(",", "").str.strip()
    df["거래금액"] = pd.to_numeric(df["거래금액"], errors="coerce")
    df["전용면적"] = pd.to_numeric(df["전용면적"], errors="coerce")
    df["층"] = pd.to_numeric(df["층"], errors="coerce")
    df["건축년도"] = pd.to_numeric(df["건축년도"], errors="coerce")

    # 평당가 (만원/m2)
    df["price_per_sqm"] = df["거래금액"] / df["전용면적"]

    # 거래년월
    df["거래년월"] = df["년"].astype(str) + df["월"].astype(str).str.zfill(2)

    # 저장
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "seoul_apt_transactions.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {output_path}")
    print(f"총 {len(df):,}건")

    return df


if __name__ == "__main__":
    df = collect_all()
    if not df.empty:
        print(f"\n{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"\n컬럼: {list(df.columns)}")
        print(f"\n샘플 5행:")
        print(df.head().to_string())
        print(f"\n결측치 현황:")
        print(df.isnull().sum().to_string())
