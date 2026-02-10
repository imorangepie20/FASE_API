# Kuka Spotify 추천 시스템

Spotify 음악 추천 API - 오디오 피처 + 텍스트 임베딩 앙상블 모델

## 모델 성능

| 모델 | 방식 | NDCG |
|------|------|------|
| **ensemble** (기본) | 오디오 40% + 텍스트 60% | **0.571** |
| knn | 오디오 피처만 | 0.498 |
| text | 텍스트 임베딩만 | 0.408 |
| hybrid | 오디오 10% + 텍스트 90% | 0.407 |

## 테스트 결과 (장르 일치율)

| 검색어 | liked 곡 수 | Ensemble | KNN |
|--------|------------|----------|-----|
| BTS | 170 | 100% (5/5) | 0% |
| BTS (k=20) | 170 | 100% (20/20) | - |
| Miles Davis | 2 | 100% (5/5) | 0% |
| Nirvana | 63 | 100% (5/5) | - |
| Taylor Swift | 15 | 60% (3/5) | 0% |

## 설치

### 1. 의존성 설치
```bash
pip install -r requirements_kuka.txt
```

### 2. main.py에 라우터 등록
```python
# main.py에 추가
from app.routers.Kuka import recommend as kuka_recommend
from app.services.Kuka.service import spotify_service

# lifespan 또는 startup 이벤트에서
spotify_service.load()

# 라우터 등록
app.include_router(kuka_recommend.router)
```

### 3. 서버 실행
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### GET /api/spotify/recommend

음악 추천 API

**파라미터:**
| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| artist | str | null | 아티스트명 (예: BTS) |
| song | str | null | 곡명 (예: Dynamite) |
| k | int | 10 | 추천 개수 (1-50) |
| model | str | ensemble | ensemble / knn / text / hybrid |
| diversity | float | 0.0 | 다양성 (0.0-1.0, knn/ensemble만) |
| explain | bool | false | RAG 설명 생성 (Gemini 필요) |

**예시:**
```bash
# BTS 좋아하는 사람에게 K-pop 추천
curl "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5"

# KNN 오디오 기반 추천
curl "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5&model=knn"

# 다양성 적용 (MMR)
curl "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5&diversity=0.3"
```

### GET /api/spotify/models

사용 가능한 모델 정보 조회

```bash
curl "http://localhost:8000/api/spotify/models"
```

## 파일 구조

```
app/
├── routers/Kuka/
│   ├── __init__.py
│   └── recommend.py      # API 엔드포인트
├── services/Kuka/
│   ├── __init__.py
│   └── service.py        # 추천 엔진 핵심 로직
└── schemas/Kuka/
    ├── __init__.py
    └── schemas.py        # Pydantic 스키마

data/
└── spotify_cleaned.parquet  # 89,740곡 메타데이터

models/
└── text_embeddings.npy      # MiniLM 임베딩 캐시 (384d)
```

## 데이터 스펙

- **곡 수**: 89,740곡
- **오디오 피처**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- **임베딩**: MiniLM-L6-v2 (384차원)
- **인덱스**: FAISS IndexFlatIP (코사인 유사도)

## 알려진 이슈

### 1. MMR 버그 (diversity > 0)
`diversity=0.3` 등 설정 시 결과가 1개만 반환되는 버그 있음.
`diversity=0` (순수 KNN)은 정상 작동.

### 2. Gemini API 키 필요
`explain=true` 사용하려면 환경변수 설정:
```bash
export GEMINI_API_KEY=your_api_key
```

### 3. 첫 부팅 시간
`text_embeddings.npy` 파일이 없으면 최초 실행 시 임베딩 생성에 시간 소요.
(포함된 npy 파일 사용 시 즉시 시작)

## 핵심 결론

1. **Ensemble이 모든 장르에서 KNN 압도** - 텍스트가 장르 신호 담당
2. **데이터 양 = 품질** - liked 곡이 많을수록 추천 정확도 상승
3. **검색은 부분 매칭** - `artist=Taylor`는 "Taylor Swift", "Teyana Taylor" 모두 포함
