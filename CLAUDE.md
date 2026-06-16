# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Commute-Value Finder** — 통근거리 기반 부동산 가치 최적화 플랫폼. 회사 위치 기준으로 통근 시간과 아파트 실거래가를 공간적으로 결합하여 저평가 입지(Blue Zone)를 발굴하고, LLM 브리핑으로 맞춤 추천을 제공한다. 상세 요구사항은 `prd.md` 참조.

## Project Structure

프로젝트 코드는 `commute_value_finder/` 디렉토리에 위치한다.

```
commute_value_finder/
├── config.py             # 설정값 중앙 관리 (회사 위치, 구 코드, 분석 기간)
├── src/
│   ├── data_collector.py # 국토부 API 데이터 수집 (동시 5개 요청)
│   └── utils.py          # load_env(), get_project_root()
├── data/                 # 수집 데이터 (CSV)
└── output/               # 지도 HTML
```

## Tech Stack

- **Language:** Python 3.14
- **Web Dashboard:** Streamlit
- **Map Visualization:** Folium (HTML 출력)
- **Spatial Analysis:** GeoPandas (행정동 경계, 공간 조인)
- **Data Processing:** Pandas / NumPy
- **Statistical Modeling:** Scikit-learn (선형 회귀, 잔차 분석)
- **LLM:** Anthropic Claude API
- **Commute Time:** 카카오맵 Directions API

## Environment Variables

`commute_value_finder/.env` 파일에 다음 키 필요 (`.gitignore` 적용됨):
- `MOLIT_API_KEY` — 공공데이터포털 실거래가 API 디코딩 인증키
- `KAKAO_API_KEY` — 카카오 REST API 키
- `ANTHROPIC_API_KEY` — Anthropic API 키

`src/utils.py`의 `load_env(required_keys=[...])` 함수로 필요한 키만 선택적 검증 가능.

## Key External APIs

- **실거래가 API:** `http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev`
  - 파라미터: `serviceKey` (소문자), `LAWD_CD`, `DEAL_YMD`, `numOfRows`, `pageNo`
  - 응답 필드: `aptNm`, `dealAmount`, `excluUseAr`, `floor`, `buildYear`, `umdNm`, `sggCd`, `dealYear`, `dealMonth`
  - `numOfRows=1000`이면 대부분 한 페이지로 조회 가능
  - 주의: `openapi.molit.go.kr:8081` 엔드포인트는 DNS가 127.0.0.1로 리졸브되어 사용 불가
- **카카오맵 Directions:** REST API, 월 300,000건 무료 한도
- **Anthropic Claude:** 프롬프트에 실거래가 데이터 기반 제약 필수 (할루시네이션 방지)

## Commands

```bash
cd commute_value_finder

# Install dependencies
python3 -m pip install -r requirements.txt

# Run data collector (MOLIT_API_KEY 필요)
python3 src/data_collector.py

# Run Streamlit dashboard (Phase 4 이후)
streamlit run app.py
```

## Architecture

4단계 파이프라인:

1. **데이터 수집** (`src/data_collector.py`) — 공공데이터포털 API에서 서울 25개 구 실거래가 수집 (ThreadPoolExecutor 병렬 처리)
2. **분석/모델링** (`zone_analyzer.py`) — Pandas 정제 → GeoPandas 공간 조인 → Scikit-learn 선형 회귀 → 잔차 분석으로 Blue/Gray/Red Zone 분류
3. **시각화** (`map_builder.py`) — Folium 히트맵, 통근 반경 원, Zone별 마커, 툴팁
4. **LLM 브리핑** (`src/llm_briefing.py`) — Blue Zone 데이터를 컨텍스트로 OpenAI API(GPT-4o)에 주입, 동네 3곳 추천
5. **대시보드** (`app.py`) — Streamlit 통합 UI (지도/분석테이블/AI브리핑/회귀분석 탭, 지하철 오버레이, 아파트 상세)

## Zone Classification Logic

통근시간(X) → 평당가(Y) 선형 회귀 후 잔차 기반 분류:
- **Blue Zone** (저평가): 잔차 < -1σ
- **Gray Zone** (적정): |잔차| < 1σ
- **Red Zone** (프리미엄): 잔차 > +1σ

## Development Phases

| Phase | 산출물 | 상태 |
|-------|--------|------|
| 0 | `src/data_collector.py`, `data/seoul_apt_transactions.csv` | 완료 |
| 1 | `src/map_builder.py`, `output/map_phase1.html` | 완료 |
| 2 | `src/zone_analyzer.py`, `output/map_phase2.html` | 완료 |
| 3 | `src/llm_briefing.py`, `output/map_final.html` | 완료 |
| 4 | `app.py` (Streamlit 대시보드), `README.md` | 완료 |
