"""Commute Value Finder - 설정값 중앙 관리"""

from datetime import datetime

# ── 회사 위치 (기준점) ──────────────────────────────────────
COMPANY_NAME = "강남구 삼성동"
COMPANY_LAT = 37.5145
COMPANY_LNG = 127.0633

# ── 분석 기간 ────────────────────────────────────────────
ANALYSIS_MONTHS = 12  # 최근 N개월

# ── 통근 반경 ───────────────────────────────────────────
COMMUTE_THRESHOLDS = [30, 60]           # 분
COMMUTE_RADIUS_M = [10_000, 20_000]     # 지도 표시용 미터

# ── 지도 기본 뷰 ───────────────────────────────────────
SEOUL_CENTER = (37.5665, 126.9780)

# ── Zone 분류 ──────────────────────────────────────────
ZONE_SIGMA = 1.0  # 잔차 ±Nσ 기준

# ── API 설정 ──────────────────────────────────────────
MAX_RETRIES = 3
MAX_WORKERS = 5
FALLBACK_SPEED = 0.3  # km/min (= 18km/h, 직선거리→통근 추정용)

# ── LLM ───────────────────────────────────────────────
LLM_MODEL = "gpt-4o"
LLM_MAX_TOKENS = 1000

# ── 서울시 25개 구 법정동코드 (LAWD_CD 앞 5자리) ────────
SEOUL_DISTRICT_CODES = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}


def get_analysis_months() -> list[str]:
    """최근 ANALYSIS_MONTHS개월의 YYYYMM 리스트 반환 (오름차순)"""
    now = datetime.now()
    months = []
    for i in range(1, ANALYSIS_MONTHS + 1):
        year = now.year
        month = now.month - i
        while month <= 0:
            month += 12
            year -= 1
        months.append(f"{year}{month:02d}")
    return sorted(months)


# ── 모델 정제 파라미터 ──
AREA_BANDS = [60, 85, 135]            # 전용면적 평형 구간 경계 (㎡)
PRICE_OUTLIER_PCT = (1, 99)           # 평당가 이상치 절단 백분위
MIN_TRANSACTIONS_PER_COMPLEX = 5      # Blue 게이트: 최소 거래건수
SHRINKAGE_K = 5                       # Empirical Bayes 수축 강도
RECENCY_MONTHS = 6                    # Blue 게이트: 최근 거래 존재 기준(개월)
