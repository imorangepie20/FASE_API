"""
M1 API Router
트랙 분석, 추천, 피드백 재학습 API
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from M1.service import M1RecommendationService

router = APIRouter(prefix="/api/m1", tags=["M1 - Audio Feature Prediction"])

# 서비스 초기화
MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_predictor.pkl")
m1_service = M1RecommendationService(model_path=MODEL_PATH)


# ==================== Request/Response Models ====================

class AnalyzeRequest(BaseModel):
    userid: int

class RecommendResponse(BaseModel):
    user_id: int
    playlist_id: int
    message: str
    recommendations: List[Dict[str, Any]]

class DeleteTrackRequest(BaseModel):
    users_id: int
    playlists_id: int
    tracks_id: int

class RetrainRequest(BaseModel):
    user_id: int
    deleted_track_ids: List[int]

class TransferEMSRequest(BaseModel):
    userid: int
    track_ids: Optional[List[int]] = None

class TransferEMSResponse(BaseModel):
    message: str
    user_id: int
    track_count: int
    ems_tracks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    gms_playlist_id: Optional[int] = None


class RandomEMSRequest(BaseModel):
    userid: int
    limit: Optional[int] = 100


class RandomEMSResponse(BaseModel):
    message: str
    user_id: int
    track_count: int
    tracks: List[Dict[str, Any]]


# ==================== API Endpoints ====================

@router.get("/health")
async def health_check():
    """M1 모듈 상태 확인"""
    return {
        "status": "healthy",
        "module": "M1 - Audio Feature Prediction",
        "model_loaded": m1_service.is_model_loaded(),
        "model_path": MODEL_PATH
    }


@router.post("/analyze")
async def analyze_user(request: AnalyzeRequest, db: Session = Depends(get_db)):
    """
    사용자 분석 및 모델 학습 (1-4단계)
    
    1. 사용자 정보 검색 → 이메일 폴더 생성
    2. 기본 모델을 사용자 폴더로 복사
    3. PMS 트랙 가져오기
    4. 모델 추가학습 → 파일명에 _ 붙임
    """
    try:
        user_id = request.userid
        
        # 1단계: 사용자 정보 조회
        user_info = m1_service.get_user_info(db, user_id)
        if not user_info:
            return {
                "success": False,
                "message": f"사용자 {user_id}를 찾을 수 없습니다.",
                "step": 1
            }
        
        email = user_info['email']
        email_prefix = m1_service.get_email_prefix(email)
        print(f"[M1] 1단계: 사용자 정보 조회 완료 - {email}")
        
        # 2단계: 사용자 폴더 생성 + 모델 복사
        user_model_path = m1_service.copy_base_model_to_user(email)
        if not user_model_path:
            return {
                "success": False,
                "message": "기본 모델 파일이 없습니다.",
                "step": 2
            }
        print(f"[M1] 2단계: 모델 복사 완료 - {user_model_path}")
        
        # 3-4단계: PMS 트랙으로 모델 추가학습
        train_result = m1_service.train_user_model(db, user_id, email)
        
        if not train_result['success']:
            return {
                "success": False,
                "message": train_result['message'],
                "step": 3
            }
        
        print(f"[M1] 3-4단계: 모델 학습 완료 - {train_result['track_count']}곡")
        
        return {
            "success": True,
            "message": f"사용자 {email_prefix} 모델 학습 완료",
            "user_id": user_id,
            "email": email,
            "email_prefix": email_prefix,
            "track_count": train_result['track_count'],
            "model_path": train_result['model_path']
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"오류 발생: {str(e)}"
        }


@router.post("/recommend/{user_id}", response_model=RecommendResponse)
async def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    """
    사용자 추천 생성 및 GMS 플레이리스트 저장
    
    1. PMS에서 사용자 선호 트랙 분석
    2. EMS에서 후보 트랙 평가
    3. final_score >= 0.7 트랙을 GMS에 저장
    4. Top 10 추천 반환
    """
    try:
        results = m1_service.get_recommendations(db, user_id)
        
        if results.empty:
            return RecommendResponse(
                user_id=user_id,
                playlist_id=0,
                message="No preferences found for this user",
                recommendations=[]
            )
        
        # GMS 플레이리스트 저장
        playlist_id = m1_service.save_gms_playlist(db, user_id, results)
        
        # Top 10 반환
        top_recommendations = results.head(10).copy()
        
        if 'final_score' in top_recommendations.columns:
            top_recommendations['score'] = top_recommendations['final_score']
        elif 'recommendation_score' in top_recommendations.columns:
            top_recommendations['score'] = top_recommendations['recommendation_score']
        
        recommendations = top_recommendations.to_dict(orient='records')
        
        return RecommendResponse(
            user_id=user_id,
            playlist_id=playlist_id,
            message="GMS playlist created successfully",
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deleted-track")
async def deleted_track(request: DeleteTrackRequest, db: Session = Depends(get_db)):
    """
    트랙 삭제 + 모델 재학습 (Spring Boot TestServiceImpl 연동)
    
    - 트랙을 플레이리스트에서 삭제
    - 삭제된 트랙을 '싫어요'로 학습
    """
    try:
        user_id = request.users_id
        playlist_id = request.playlists_id
        track_id = request.tracks_id
        
        # 1. 트랙 삭제
        delete_result = m1_service.delete_tracks_from_playlist(db, playlist_id, [track_id])
        
        # 2. 모델 재학습
        retrain_result = m1_service.retrain_with_feedback(db, user_id, [track_id])
        
        return {
            "message": f"트랙 {track_id} 삭제 및 모델 재학습 완료.",
            "retrain_metrics": retrain_result.get("metrics", {})
        }
    except Exception as e:
        return {"message": f"오류 발생: {str(e)}"}


@router.delete("/playlist/{playlist_id}/tracks")
async def delete_tracks(
    playlist_id: int, 
    track_ids: List[int] = Query(..., description="삭제할 트랙 ID 목록"),
    db: Session = Depends(get_db)
):
    """GMS 플레이리스트에서 트랙 삭제"""
    try:
        result = m1_service.delete_tracks_from_playlist(db, playlist_id, track_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain/{user_id}")
async def retrain_model(
    user_id: int,
    request: RetrainRequest,
    db: Session = Depends(get_db)
):
    """
    피드백 기반 모델 재학습
    
    - 삭제된 트랙들을 '싫어요' 데이터로 활용
    - PreferenceClassifier 재학습
    """
    try:
        result = m1_service.retrain_with_feedback(db, user_id, request.deleted_track_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    """
    사용자 음악 취향 프로필 조회
    
    - PMS 데이터 기반 분석 결과 반환
    """
    try:
        profile = m1_service.get_user_profile(db, user_id)
        return {
            "user_id": user_id,
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/random-ems", response_model=RandomEMSResponse)
async def get_random_ems_tracks(request: RandomEMSRequest, db: Session = Depends(get_db)):
    """
    EMS 전체에서 랜덤 트랙 추출 (5-1단계용)
    
    - DB에서 직접 SQL RAND()로 균등 분포 랜덤 추출
    - 중복 제거 (track_id 기준 DISTINCT)
    - 아티스트별 편중 없이 다양한 트랙 반환
    """
    try:
        user_id = request.userid
        limit = request.limit or 100
        
        # DB에서 직접 랜덤 100곡 추출
        random_tracks_df = m1_service.get_random_ems_tracks(db, limit)
        
        if random_tracks_df.empty:
            return RandomEMSResponse(
                message="EMS에 트랙이 없습니다.",
                user_id=user_id,
                track_count=0,
                tracks=[]
            )
        
        tracks_list = random_tracks_df.to_dict(orient='records')
        
        return RandomEMSResponse(
            message=f"EMS에서 랜덤 {len(tracks_list)}곡 추출 완료",
            user_id=user_id,
            track_count=len(tracks_list),
            tracks=tracks_list
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transfer-ems", response_model=TransferEMSResponse)
async def transfer_ems_tracks(request: TransferEMSRequest, db: Session = Depends(get_db)):
    """
    EMS 트랙 전달 및 추천 생성 (5-8단계)
    
    5. 전달받은 track_ids로 EMS 트랙 조회
    6. 사용자 추가학습 모델(email_.pkl)로 평가
    7. 평가 결과 반환
    8. 결과 박스에 출력
    """
    try:
        user_id = request.userid
        track_ids = request.track_ids
        
        # 사용자 정보 조회 (추가학습 모델 경로 확인용)
        user_info = m1_service.get_user_info(db, user_id)
        if not user_info:
            return TransferEMSResponse(
                message=f"사용자 {user_id}를 찾을 수 없습니다.",
                user_id=user_id,
                track_count=0,
                ems_tracks=[],
                recommendations=[]
            )
        
        email = user_info['email']
        
        # 5단계: EMS 트랙 조회
        if track_ids and len(track_ids) > 0:
            ems_tracks_df = m1_service.get_tracks_by_ids(db, track_ids)
        else:
            ems_tracks_df = m1_service.get_ems_tracks_from_db(db, user_id)
        
        if ems_tracks_df.empty:
            return TransferEMSResponse(
                message="EMS에 트랙이 없습니다.",
                user_id=user_id,
                track_count=0,
                ems_tracks=[],
                recommendations=[]
            )
        
        ems_tracks_list = ems_tracks_df.to_dict(orient='records')
        
        # 6단계: 사용자 추가학습 모델로 평가
        results = m1_service.evaluate_with_user_model(db, user_id, email, ems_tracks_df)
        
        if results.empty:
            return TransferEMSResponse(
                message=f"EMS 트랙 {len(ems_tracks_df)}곡 조회 완료. 추가학습 모델이 없어 평가 불가.",
                user_id=user_id,
                track_count=len(ems_tracks_df),
                ems_tracks=ems_tracks_list,
                recommendations=[]
            )
        
        # 7단계: GMS 품질 필터링 (0.7 이상만 저장)
        gms_pass = results[results['score'] >= 0.7].sort_values(
            by='score', ascending=False
        ).copy()
        gms_pass['recommendation_score'] = gms_pass['score']

        gms_playlist_id = None

        # GMS 플레이리스트에 통과한 트랙이 있으면 생성
        if not gms_pass.empty:
            gms_playlist_id = m1_service.save_gms_playlist(db, user_id, gms_pass)

        # 8단계: 결과 반환
        top_recommendations = results.head(20).copy()
        top_recommendations['score'] = top_recommendations['score']
        recommendations_list = top_recommendations.to_dict(orient='records')

        return TransferEMSResponse(
            message=f"EMS 트랙 {len(ems_tracks_df)}곡 평가 완료. {len(results)}곡 추천됨. {len(gms_pass)}곡 GMS 통과.",
            user_id=user_id,
            track_count=len(ems_tracks_df),
            ems_tracks=ems_tracks_list,
            recommendations=recommendations_list,
            gms_playlist_id=gms_playlist_id
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
