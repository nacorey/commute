"""대시보드 표시 로직: 라벨·리스크 플래그·외부 링크·후보 랭킹 (순수함수)."""

from urllib.parse import quote

ZONE_LABELS = {
    "Blue": "저평가 후보 (확인 필요)",
    "Gray": "적정",
    "Red": "프리미엄",
}


def zone_label(zone: str) -> str:
    return ZONE_LABELS.get(zone, zone)


def risk_flags(row) -> list:
    """단지의 '왜 싸 보이는지' 자동 리스크 플래그."""
    flags = []
    dev = row.get("deviation_pct")
    if dev is not None and abs(dev) > 50:
        flags.append("극단 편차 — 이상치/데이터 확인 필요")
    n = row.get("n")
    if n is not None and n < 5:
        flags.append("거래 적음 — 표본 신중")
    if row.get("confidence") == "낮음":
        flags.append("표본 신뢰 낮음")
    sd = row.get("subway_dist_km")
    if sd is not None and sd > 1.0:
        flags.append("역세권 아님 — 도보 먼 편")
    return flags


def naver_land_url(gu: str, dong: str, apt: str) -> str:
    """네이버 지도검색 링크 (현재 매물·호가 확인용).

    매물 전용 경로(m.land)는 단지 매칭 실패 시 빈 화면이 뜨므로,
    이름 검색이 안정적이고 단지 장소 카드의 '부동산' 탭(시세·매물)으로
    이어지는 지도검색으로 보낸다. 한글·공백·특수문자는 URL 인코딩한다.
    """
    keyword = quote(f"{dong} {apt}".strip())
    return f"https://map.naver.com/p/search/{keyword}"


def hogangnono_url(dong: str, apt: str) -> str:
    """호갱노노 검색 링크 (시세·실거래 확인용).

    동을 함께 넣어 동명이 단지의 오매칭을 줄인다. URL 인코딩 적용.
    """
    keyword = quote(f"{dong} {apt}".strip())
    return f"https://hogangnono.com/search?q={keyword}"


def rank_candidates(zones, max_commute, max_price, top_n=20):
    """Blue 후보를 조건으로 필터 → 저평가순(잔차 오름차순) 정렬."""
    df = zones[
        (zones["zone"] == "Blue")
        & (zones["commute_minutes"] <= max_commute)
        & (zones["avg_price_per_sqm"] <= max_price)
    ].copy()
    return df.sort_values("final_resid").head(top_n)
