"""Spotify 추천 API 엔드포인트 (Kuka)"""

from fastapi import APIRouter, HTTPException, Query
from app.services.Kuka.service import spotify_service
from app.schemas.Kuka.schemas import RecommendResponse, TrackInfo, ModelInfoResponse

router = APIRouter(prefix="/api/spotify", tags=["spotify-recommendation"])


@router.on_event("startup")
async def startup():
    """서버 시작 시 데이터 로딩"""
    spotify_service.load()


@router.get("/recommend", response_model=RecommendResponse)
async def recommend(
    artist: str = Query(None, description="아티스트명 (예: BTS)"),
    song: str = Query(None, description="곡명 (예: Dynamite)"),
    k: int = Query(10, ge=1, le=50, description="추천 개수"),
    model: str = Query("ensemble", description="추천 모델: ensemble / knn / text / hybrid"),
    diversity: float = Query(0.0, ge=0.0, le=1.0, description="다양성 (0.0=순수추천, 0.3=적당, 0.5=높음)"),
    explain: bool = Query(False, description="RAG 설명 생성 여부"),
):
    """
    음악 추천 API

    - **model=ensemble**: 오디오+텍스트 앙상블 α=0.4 (NDCG=0.571, 챔피언)
    - **model=knn**: 오디오 피처 KNN (NDCG=0.498)
    - **model=text**: 텍스트 임베딩 FAISS (NDCG=0.408)
    - **model=hybrid**: 오디오+텍스트 결합 α=0.1 (NDCG=0.407)
    - **diversity**: 0.0이면 순수 추천, 0.3이면 MMR(λ=0.7) 적용 (knn/ensemble)
    - **explain**: true면 Gemini RAG로 추천 이유 설명 (수치 인용 포함)
    """
    if not artist and not song:
        raise HTTPException(status_code=400, detail="artist 또는 song 중 하나는 필수입니다")

    # 좋아하는 곡 검색
    liked_indices = spotify_service.find_tracks(artist=artist, song=song)
    if not liked_indices:
        raise HTTPException(
            status_code=404,
            detail=f"'{artist or ''} {song or ''}' 검색 결과 없음"
        )

    query = f"{artist or ''} {song or ''}".strip()

    # 추천 실행
    if model == "ensemble":
        results = spotify_service.recommend_hybrid(liked_indices, k=k, alpha=0.4)
    elif model == "knn":
        results = spotify_service.recommend_knn(liked_indices, k=k, diversity=diversity)
    elif model == "text":
        results = spotify_service.recommend_text(liked_indices, k=k)
    elif model == "hybrid":
        results = spotify_service.recommend_hybrid(liked_indices, k=k, alpha=0.1)
    else:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 모델: {model}")

    # RAG 설명 생성
    explanation = None
    if explain:
        explanation = spotify_service.generate_explanation(liked_indices, results)

    return RecommendResponse(
        model=model,
        query=query,
        total_tracks=len(liked_indices),
        recommendations=[TrackInfo(**{k: v for k, v in r.items() if k != 'index'}) for r in results],
        explanation=explanation,
        diversity=diversity,
    )


@router.get("/models", response_model=ModelInfoResponse)
async def get_models():
    """사용 가능한 추천 모델 정보"""
    return spotify_service.get_model_info()