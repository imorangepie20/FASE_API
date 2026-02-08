"""
M3 API Router
CatBoost 기반 협업 필터링 추천 시스템

주요 기능:
- PMS 플레이리스트 분석
- CatBoost 모델로 사용자 취향 벡터 생성
- 유클리드 거리 기반 EMS 트랙 추천
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys
import glob

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter(prefix="/api/m3", tags=["M3 - CatBoost Recommender"])

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

# ==================== Request/Response Models ====================

class AnalysisRequest(BaseModel):
    """분석 요청"""
    userid: int = Field(..., description="사용자 ID")


class AnalysisResponse(BaseModel):
    """분석 응답"""
    status: str
    message: str
    user_id: Optional[int] = None
    model_path: Optional[str] = None
    recommendations_count: Optional[int] = None


class RecommendRequest(BaseModel):
    """추천 요청"""
    user_id: int
    top_k: int = Field(50, ge=1, le=100, description="추천 트랙 수")


class TrackRecommendation(BaseModel):
    """추천 트랙"""
    track_id: str
    track_name: str
    artist: str
    album_name: str
    genre: str
    distance: float = Field(..., description="사용자 취향과의 거리 (낮을수록 유사)")


class RecommendResponse(BaseModel):
    """추천 응답"""
    status: str
    user_id: int
    model_used: str
    recommendations: List[TrackRecommendation]


# ==================== Helper Functions ====================

def get_latest_model_path(user_id: int) -> Optional[str]:
    """사용자의 최신 모델 파일 경로 조회"""
    search_pattern = os.path.join(MODEL_DIR, f"recommender_U{user_id}_*.cbm")
    model_files = glob.glob(search_pattern)
    
    if not model_files:
        return None
    
    # 가장 최근 수정된 파일 반환
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def get_any_model_path() -> Optional[str]:
    """아무 모델 파일이나 반환 (기본 모델용)"""
    search_pattern = os.path.join(MODEL_DIR, "*.cbm")
    model_files = glob.glob(search_pattern)
    
    if not model_files:
        return None
    
    return model_files[0]


# ==================== API Endpoints ====================

@router.get("/health")
async def health_check():
    """M3 모듈 상태 확인"""
    # .cbm 파일 존재 확인
    cbm_files = glob.glob(os.path.join(MODEL_DIR, "*.cbm"))
    model_exists = len(cbm_files) > 0
    
    return {
        "status": "healthy" if model_exists else "degraded",
        "module": "M3 - CatBoost Recommender",
        "model_loaded": model_exists,
        "model_count": len(cbm_files),
        "model_dir": MODEL_DIR,
        "features": {
            "algorithm": "CatBoost + Collaborative Filtering",
            "target_columns": ["danceability", "energy", "key", "loudness", "mode", 
                              "speechiness", "acousticness", "instrumentalness", "liveness"],
            "distance_metric": "Euclidean"
        }
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_user(request: AnalysisRequest):
    """
    사용자 분석 및 추천 생성
    
    1. PMS 플레이리스트 조회
    2. 기존 모델 로드 또는 새 모델 학습
    3. PMS 트랙으로 사용자 취향 벡터 생성
    4. EMS에서 유사 트랙 찾아 GMS에 저장
    """
    try:
        user_id = request.userid
        
        # 모델 확인
        model_path = get_latest_model_path(user_id)
        
        if not model_path:
            # 기본 모델 사용
            model_path = get_any_model_path()
            if not model_path:
                return AnalysisResponse(
                    status="error",
                    message="모델 파일을 찾을 수 없습니다. 먼저 학습이 필요합니다."
                )
        
        # TODO: 실제 분석 및 추천 로직 구현
        # m3.py의 process_analysis 함수 연동
        
        return AnalysisResponse(
            status="success",
            message=f"사용자 {user_id} 분석 및 GMS 생성 완료",
            user_id=user_id,
            model_path=os.path.basename(model_path),
            recommendations_count=50
        )
        
    except Exception as e:
        return AnalysisResponse(
            status="error",
            message=f"분석 중 오류 발생: {str(e)}"
        )


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    추천 트랙 조회
    
    1. 사용자 모델 로드
    2. PMS 분석하여 취향 벡터 생성
    3. EMS 트랙과 유클리드 거리 계산
    4. 가장 가까운 top_k개 반환
    """
    try:
        user_id = request.user_id
        top_k = request.top_k
        
        # 모델 확인
        model_path = get_latest_model_path(user_id)
        model_name = os.path.basename(model_path) if model_path else "default_model"
        
        # TODO: 실제 추천 로직 구현
        # 더미 응답
        recommendations = [
            TrackRecommendation(
                track_id=f"track_{i}",
                track_name=f"Recommended Track {i}",
                artist=f"Artist {i}",
                album_name=f"Album {i}",
                genre="pop",
                distance=0.1 * i
            )
            for i in range(1, min(top_k + 1, 11))
        ]
        
        return RecommendResponse(
            status="success",
            user_id=user_id,
            model_used=model_name,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{user_id}")
async def train_model(
    user_id: int,
    playlist_title: str = Query("MyPlaylist", description="플레이리스트 제목")
):
    """
    사용자 CatBoost 모델 학습
    
    1. 데이터셋 로드 (dataset.csv)
    2. CatBoost 모델 학습 (MultiRMSE)
    3. 모델 저장: recommender_U{user_id}_{title}_{date}.cbm
    """
    try:
        # TODO: 실제 학습 로직 구현
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        safe_title = "".join(x for x in playlist_title if x.isalnum())[:10]
        model_filename = f"recommender_U{user_id}_{safe_title}_{date_str}.cbm"
        
        return {
            "success": True,
            "message": f"사용자 {user_id} 모델 학습 완료",
            "user_id": user_id,
            "model_path": model_filename,
            "metrics": {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "MultiRMSE"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """저장된 모델 목록 조회"""
    try:
        cbm_files = glob.glob(os.path.join(MODEL_DIR, "*.cbm"))
        
        models = []
        for path in cbm_files:
            filename = os.path.basename(path)
            stat = os.stat(path)
            models.append({
                "filename": filename,
                "path": path,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            })
        
        # 수정 시간 기준 정렬
        models.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "total": len(models),
            "models": models
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{user_id}")
async def delete_user_models(user_id: int):
    """사용자 모델 삭제"""
    try:
        search_pattern = os.path.join(MODEL_DIR, f"recommender_U{user_id}_*.cbm")
        model_files = glob.glob(search_pattern)
        
        deleted_count = 0
        for path in model_files:
            os.remove(path)
            deleted_count += 1
        
        return {
            "success": True,
            "message": f"사용자 {user_id}의 모델 {deleted_count}개 삭제 완료",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
