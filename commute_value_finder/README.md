# Commute-Value Finder

통근·지하철 접근성 대비 **가격이 낮은 아파트 단지를 발굴하는 스크리닝 도구**입니다. 서울시 아파트 실거래가를 2단계 헤도닉 모델로 분석해, 단위 특성(면적·층·연식)과 시점을 통제한 뒤 "통근·지하철 대비 저평가된 단지 후보"를 단지 단위로 찾아냅니다.

> ⚠️ **이것은 투자 예측기가 아니라 스크리너입니다.** 백테스트 결과, 저평가 후보가 향후 가격 상승을 예측하지는 못했습니다(통근·지하철 대비 *지속적으로* 싼 곳을 찾는 용도). 잔차에 공간 자기상관도 유의(Moran's I≈0.57)해 학군 등 누락 요인의 영향이 섞일 수 있습니다. **모든 후보는 직접 확인**하세요.

## 아키텍처

```
실거래가(76,986건)
  └─ preprocessor   정제·검증·중복제거·이상치·취소거래 필터
  └─ value_model    [1단계] log(평당가) ~ 면적·평형·층·연식·시점  → 거래 잔차
                    [집계]  단지별 잔차 + Empirical Bayes 축소추정 → 입지가치지수
                    [2단계] 입지가치지수 ~ 통근시간 + 지하철거리   → 최종 잔차
                    [분류]  ±1σ + Blue 신뢰 게이트(거래건수·최근성·신뢰도)
  └─ commute        통근 인터페이스(Phase A: 카카오 운전, Phase B: GTFS 교체용)
  └─ subway_access  단지별 최근접 지하철역 도보거리
  └─ apt_geocoder   단지 단위 지오코딩(카카오 키워드, 캐시 + 동 좌표 폴백)
  └─ validation     시간 홀드아웃 백테스트 + Moran's I 공간 자기상관
  └─ briefing       OpenAI gpt-5.4-mini 구조화 JSON 브리핑(+규칙기반 폴백)
  └─ app.py         Streamlit 대시보드(지도·후보 랭킹·AI 브리핑·모델/검증)
```

## 설치

```bash
pip install -r requirements.txt
```
> Windows에서 `python`이 안 잡히면 `py`를 사용하세요 (예: `py run_model.py`).

## API 키 설정

`.env` 파일에 설정 (`.gitignore` 적용됨):

```
MOLIT_API_KEY=공공데이터포털_디코딩_인증키
KAKAO_API_KEY=카카오_REST_API_키
OPENAI_API_KEY=sk-...
```
- **MOLIT_API_KEY**: [공공데이터포털](https://www.data.go.kr/data/15126469/openapi.do) — 국토부 아파트 매매 실거래가 자료 활용신청
- **KAKAO_API_KEY**: [카카오 개발자](https://developers.kakao.com/) — 로컬(주소/키워드) + 카카오내비 활성화
- **OPENAI_API_KEY**: [OpenAI Platform](https://platform.openai.com/) — LLM 브리핑(유일한 유료 항목)

## 실행

### 1) 캐시 준비 (최초 1회 또는 회사 위치·데이터 변경 시)
회사 위치는 `config.py`의 `COMPANY_LAT/LNG`에서 설정합니다.

```bash
python src/data_collector.py   # 실거래가 → data/seoul_apt_transactions.csv
python src/map_builder.py       # 법정동 지오코딩 → data/geocode_cache.json
python src/zone_analyzer.py      # 동별 통근시간 → data/commute_cache.json
```
> `map_builder.py`·`zone_analyzer.py`는 옛 분석/지도 기능도 갖고 있으나, 현재 파이프라인에서는 **위 두 캐시를 빌드하는 데이터 준비 단계**로 사용합니다.

### 2) 분석 파이프라인
```bash
python run_model.py            # 정제→헤도닉→집계→지오코딩→지하철→통근→분류
                               # → data/complex_zones.csv (단지 단위 zone)
```

### 3) 검증 (선택)
```bash
python validate_and_brief.py   # 백테스트 + Moran's I + LLM 브리핑 스모크
```

### 4) 대시보드
```bash
streamlit run app.py
```

### 테스트
```bash
python -m pytest tests/
```

## 주요 기능

- **2단계 헤도닉 가치 모델** — 단위 특성·평형·시점을 통제해 구성효과를 제거하고, 통근·지하철 접근성 대비 순수 입지 저평가를 추출
- **단지 단위 해상도** — 5,000여 개 아파트 단지를 개별 분석(동 단위 평균이 아님)
- **Blue 신뢰 게이트** — 잔차 + 거래건수 + 최근성 + 신뢰도를 함께 만족해야 "후보"로 확정
- **자기검증** — 시간 홀드아웃 백테스트로 신호 유효성 점검, Moran's I로 공간 자기상관(군집 아티팩트) 검정
- **정직성 레이어** — "저평가 후보(확인 필요)" 재프레이밍, 리스크 자동 플래그(극단 편차·거래 희박·역세권 아님), 불확실성 표시
- **행동 레이어** — 예산·통근 기준 후보 랭킹 + 네이버부동산/호갱노노 링크아웃 + GTX·신설노선 교통호재 오버레이
- **AI 브리핑** — gpt-5.4-mini 구조화 JSON 출력으로 데이터 근거를 강제(할루시네이션 방지), 실패 시 규칙기반 폴백

## 데이터 출처 / 비용

| 데이터 | 출처 | 비용 |
|--------|------|------|
| 아파트 실거래가 | [공공데이터포털](https://www.data.go.kr) | 무료 |
| 통근 시간(운전) | 카카오 모빌리티 API | 무료 (월 300,000건) |
| 지오코딩 | 카카오 로컬 API | 무료 |
| 지하철역 좌표 | 보유 데이터(`data/subway_stations.json`) | 무료 |
| AI 브리핑 | OpenAI gpt-5.4-mini | 종량제(유일한 유료) |

## 모듈 구성

| 파일 | 역할 |
|------|------|
| `src/preprocessor.py` | 데이터 정제·검증 레이어 |
| `src/value_model.py` | 2단계 헤도닉 모델·단지 집계·zone 분류 |
| `src/commute.py` | 통근 추정 인터페이스(Phase A 운전) |
| `src/subway_access.py` | 지하철 접근성 피처 |
| `src/apt_geocoder.py` | 단지 지오코딩(캐시·폴백) |
| `src/validation.py` | 백테스트 + Moran's I |
| `src/briefing.py` | LLM 구조화 JSON 브리핑 |
| `src/dashboard_logic.py` | 라벨·리스크 플래그·랭킹·링크아웃 |
| `src/transit_projects.py` | 교통호재(GTX·신설) 데이터·거리 |
| `src/data_collector.py` | 실거래가 수집 |
| `src/map_builder.py` · `src/zone_analyzer.py` | (데이터 준비) 동 지오코딩·통근 캐시 빌더 |
| `run_model.py` | 엔드투엔드 분석 파이프라인 |
| `validate_and_brief.py` | 검증·브리핑 스모크 |
| `app.py` | Streamlit 대시보드 |

## 한계 (정직하게)

- "저평가"는 **미스프라이싱 확정이 아니라 조사 후보**입니다. 잔차에는 학군·향·소음·재건축 가능성 등 통제하지 못한 요인이 섞입니다.
- 통근시간은 현재 **자동차 운전 기준**(프록시)입니다. 진짜 대중교통(버스+지하철) 시간은 Phase B(GTFS)에서 동일 인터페이스로 교체 예정입니다.
- 학군·세대수·향 등은 데이터에 없어 **체크리스트로 직접 확인**하도록 안내합니다.

## 라이선스

학습 및 연구 목적으로 제작되었습니다.
