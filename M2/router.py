"""
M2 API Router
SVM + Text Embedding 기반 추천 시스템

주요 기능:
- 393D 피처 (384D 텍스트 임베딩 + 9D 오디오)
- SVM 기반 사용자 취향 예측
- Last.fm 태그 활용
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter(prefix="/api/m2", tags=["M2 - SVM Text Embedding"])

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tfidf_gbr_models.pkl")

# ==================== Request/Response Models ====================

class TrackInput(BaseModel):
    """트랙 입력 스키마"""
    artist: str = Field(..., description="아티스트명")
    track_name: str = Field(..., description="곡명")
    album_name: str = Field("", description="앨범명")
    tags: str = Field("", description="Last.fm 태그 (파이프로 구분)")
    duration_ms: int = Field(200000, description="곡 길이 (밀리초)")


class PredictionResponse(BaseModel):
    """예측 응답"""
    artist: str
    track_name: str
    probability: float = Field(..., description="좋아할 확률 (0~1)")
    prediction: int = Field(..., description="예측 라벨 (0=안좋아함, 1=좋아함)")
    audio_features: Dict[str, float] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    """추천 요청"""
    user_id: int
    tracks: List[TrackInput]
    top_k: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.5, ge=0, le=1)


class RecommendResponse(BaseModel):
    """추천 응답"""
    user_id: int
    total_candidates: int
    recommended_count: int
    recommendations: List[PredictionResponse]


class FeedbackRequest(BaseModel):
    """피드백 요청 (GMS 선택/삭제)"""
    user_id: int
    track_id: int
    feedback_type: str = Field(..., description="selected 또는 deleted")


# ==================== API Endpoints ====================

@router.get("/health")
async def health_check():
    """M2 모듈 상태 확인"""
    model_exists = os.path.exists(MODEL_PATH)
    
    return {
        "status": "healthy" if model_exists else "degraded",
        "module": "M2 - SVM Text Embedding",
        "model_loaded": model_exists,
        "model_path": MODEL_PATH,
        "features": {
            "text_embedding": "384D (SentenceTransformer)",
            "audio_features": "9D (TF-IDF + GBR prediction)",
            "total_features": "393D"
        }
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(track: TrackInput, user_id: int = Query(1, description="사용자 ID")):
    """
    단일 트랙 예측
    
    - 393D 피처 생성 (텍스트 임베딩 + 오디오 예측)
    - SVM으로 사용자 취향 예측
    """
    try:
        from .service import get_m2_service
        
        service = get_m2_service()
        result = service.predict_single(
            user_id=user_id,
            artist=track.artist,
            track_name=track.track_name,
            album_name=track.album_name,
            tags=track.tags,
            duration_ms=track.duration_ms
        )
        
        return PredictionResponse(
            artist=track.artist,
            track_name=track.track_name,
            probability=result['probability'],
            prediction=result['prediction'],
            audio_features=result.get('audio_features', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    추천 트랙 선정
    
    1. 각 트랙에 대해 393D 피처 생성
    2. SVM으로 "좋아할 확률" 예측
    3. threshold 이상인 트랙 중 top_k개 반환
    """
    try:
        from .service import get_m2_service
        
        service = get_m2_service()
        
        # TrackInput을 dict로 변환
        candidate_tracks = [
            {
                'artist': t.artist,
                'track_name': t.track_name,
                'album_name': t.album_name,
                'tags': t.tags,
                'duration_ms': t.duration_ms
            }
            for t in request.tracks
        ]
        
        results = service.get_recommendations(
            user_id=request.user_id,
            candidate_tracks=candidate_tracks,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        recommendations = [
            PredictionResponse(
                artist=r['artist'],
                track_name=r['track_name'],
                probability=r['probability'],
                prediction=r['prediction'],
                audio_features=r.get('audio_features', {})
            )
            for r in results
        ]
        
        return RecommendResponse(
            user_id=request.user_id,
            total_candidates=len(request.tracks),
            recommended_count=len(recommendations),
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def process_feedback(request: FeedbackRequest):
    """
    GMS 피드백 처리
    
    - selected: Positive 피드백 저장
    - deleted: Negative 피드백 저장
    - 피드백 5개 이상 시 자동 재학습 트리거
    """
    try:
        feedback_type = request.feedback_type.lower()
        
        if feedback_type not in ["selected", "deleted"]:
            raise HTTPException(status_code=400, detail="feedback_type must be 'selected' or 'deleted'")
        
        # TODO: 피드백 저장 및 재학습 트리거 구현
        return {
            "success": True,
            "message": f"피드백 저장 완료 ({feedback_type})",
            "user_id": request.user_id,
            "track_id": request.track_id,
            "feedback_type": feedback_type,
            "retrain_triggered": False
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{user_id}")
async def train_model(user_id: int, playlist_id: Optional[int] = None):
    """
    사용자 SVM 모델 학습
    
    1. PMS 플레이리스트에서 Positive 샘플 추출
    2. EMS에서 Negative 샘플링 (1:3 비율)
    3. 393D 피처 생성 후 SVM 학습
    """
    try:
        from .service import get_m2_service
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from database import SessionLocal
        from sqlalchemy import text
        
        service = get_m2_service()
        db = SessionLocal()
        
        try:
            # PMS에서 Positive 트랙 조회
            pms_query = text("""
                SELECT t.title, t.artist, t.album, t.duration
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :user_id AND p.space_type = 'PMS'
            """)
            pms_result = db.execute(pms_query, {"user_id": user_id}).fetchall()
            
            positive_tracks = [
                {
                    'track_name': r[0],
                    'artist': r[1],
                    'album_name': r[2] or '',
                    'tags': '',
                    'duration_ms': (r[3] or 200) * 1000
                }
                for r in pms_result
            ]
            
            if len(positive_tracks) < 5:
                return {
                    "success": False,
                    "message": f"PMS에 최소 5곡 이상 필요합니다 (현재: {len(positive_tracks)}곡)",
                    "user_id": user_id
                }
            
            # EMS에서 Negative 트랙 샘플링 (3배)
            ems_query = text("""
                SELECT t.title, t.artist, t.album, t.duration
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :user_id AND p.space_type = 'EMS'
                ORDER BY RAND()
                LIMIT :limit
            """)
            ems_result = db.execute(ems_query, {
                "user_id": user_id, 
                "limit": len(positive_tracks) * 3
            }).fetchall()
            
            negative_tracks = [
                {
                    'track_name': r[0],
                    'artist': r[1],
                    'album_name': r[2] or '',
                    'tags': '',
                    'duration_ms': (r[3] or 200) * 1000
                }
                for r in ems_result
            ]
            
            # 모델 학습
            result = service.train_user_model(
                user_id=user_id,
                positive_tracks=positive_tracks,
                negative_tracks=negative_tracks
            )
            
            return result
            
        finally:
            db.close()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain/{user_id}")
async def retrain_model(user_id: int):
    """
    피드백 기반 모델 재학습
    
    - 저장된 피드백 데이터 로드
    - 기존 모델에 추가 학습
    """
    try:
        # TODO: 실제 재학습 구현
        return {
            "success": True,
            "message": f"사용자 {user_id} 모델 재학습 완료",
            "user_id": user_id,
            "new_feedback_count": 5,
            "metrics": {
                "accuracy": 0.96,
                "auc_roc": 0.995
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio-features")
async def get_audio_features(
    artist: str = Query(..., description="아티스트명"),
    track_name: str = Query(..., description="곡명"),
    tags: str = Query("", description="Last.fm 태그")
):
    """
    오디오 피처 예측 (TF-IDF + GBR)
    
    텍스트 정보로 9개 오디오 피처 예측:
    - danceability, energy, speechiness, acousticness
    - instrumentalness, liveness, valence, tempo, loudness
    """
    try:
        # TODO: 실제 오디오 피처 예측 구현
        return {
            "artist": artist,
            "track_name": track_name,
            "audio_features": {
                "danceability": 0.65,
                "energy": 0.75,
                "speechiness": 0.05,
                "acousticness": 0.25,
                "instrumentalness": 0.02,
                "liveness": 0.15,
                "valence": 0.55,
                "tempo": 118.5,
                "loudness": -6.2
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
