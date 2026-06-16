# Commute-Value Finder

통근거리 기반 부동산 가치 최적화 플랫폼. 회사 위치를 기준으로 서울시 아파트 실거래가와 통근 시간을 공간적으로 결합하여, 통근 효율 대비 저평가된 'Blue Zone'을 발굴하고 AI 브리핑으로 맞춤 추천을 제공합니다.

## 설치

```bash
pip install -r requirements.txt
```

## API 키 설정

`.env` 파일에 다음 키를 설정하세요:

```
MOLIT_API_KEY=공공데이터포털_디코딩_인증키
KAKAO_API_KEY=카카오_REST_API_키
OPENAI_API_KEY=sk-...
```

- **MOLIT_API_KEY**: [공공데이터포털](https://www.data.go.kr/data/15126469/openapi.do)에서 '국토교통부_아파트 매매 실거래가 자료' 활용신청
- **KAKAO_API_KEY**: [카카오 개발자](https://developers.kakao.com/)에서 REST API 키 발급, 지도/로컬 + 카카오내비 서비스 활성화
- **OPENAI_API_KEY**: [OpenAI Platform](https://platform.openai.com/)에서 API 키 발급

## 실행

```bash
# 전체 분석 파이프라인 (Phase 0~3)
python main.py

# 웹 대시보드
streamlit run app.py
```

## 주요 기능

### 1. 실거래가 데이터 수집
서울시 25개 구 최근 12개월 아파트 매매 실거래가를 공공데이터포털 API로 수집

### 2. 인터랙티브 히트맵
법정동별 평당가를 Folium 히트맵으로 시각화, 통근 반경 원 오버레이

### 3. Zone 분류 (잔차 분석)
통근시간→평당가 선형 회귀 후 잔차 ±1σ 기준으로 Blue(저평가)/Gray(적정)/Red(프리미엄) 분류

### 4. AI 브리핑
GPT-4o 기반 맞춤형 동네 추천 (예산, 통근 상한, 우선순위 반영)

### 5. Streamlit 대시보드
슬라이더 필터, Zone 토글, 지하철 노선 오버레이, 아파트 상세 조회, CSV 다운로드

## 데이터 출처

| 데이터 | 출처 | 비용 |
|--------|------|------|
| 아파트 실거래가 | [공공데이터포털](https://www.data.go.kr) | 무료 |
| 통근 시간 | 카카오 모빌리티 API | 무료 (월 300,000건) |
| 지오코딩 | 카카오 로컬 API / Nominatim | 무료 |
| AI 브리핑 | OpenAI API (GPT-4o) | 종량제 |

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다.
