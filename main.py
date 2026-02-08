"""
AI Music Analysis API
EMS (Explore Music Space) 데이터 기반 AI 분석 서버

Models:
- M1: Audio Feature Prediction (오디오 특성 예측 + 하이브리드 추천)
- M2: Content-based Recommendation (TF-IDF + GBR)
- M3: Collaborative Filtering (CatBoost)
"""
from fastapi import FastAPI, HTTPException, Query
import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
import os
import sys
import numpy as np

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== Lifespan (시작/종료 이벤트) ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시
    print("=" * 60)
    print("🚀 AI Music Analysis API 시작")
    print("=" * 60)
    
    # DB 연결 테스트
    try:
        from database import test_connection
        if test_connection():
            print("✅ Database 연결 성공")
        else:
            print("⚠️ Database 연결 실패 - API는 계속 실행됨")
    except Exception as e:
        print(f"⚠️ Database 모듈 로드 실패: {e}")
    
    # M1 모델 상태
    try:
        model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
        if os.path.exists(model_path):
            print(f"✅ M1 모델 파일 존재: {model_path}")
        else:
            print(f"⚠️ M1 모델 파일 없음: {model_path}")
    except Exception as e:
        print(f"⚠️ M1 모델 확인 실패: {e}")
    
    print("=" * 60)
    
    yield  # 앱 실행
    
    # 종료 시
    print("🛑 AI Music Analysis API 종료")


# ==================== FastAPI 앱 초기화 ====================

app = FastAPI(
    title="AI Music Analysis API",
    description="""
## EMS 데이터 기반 음악 분석 및 추천 API

### 모델
- **M1**: 오디오 특성 예측 + 하이브리드 추천
- **M2**: TF-IDF 콘텐츠 기반 추천 (예정)
- **M3**: CatBoost 협업 필터링 (예정)

### 연동
- Node.js Backend EMS API와 통신
- Spring Boot Backend와 연동
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8080",  # Spring Boot
        "https://homological-ashlyn-supercrowned.ngrok-free.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== M1 Router 등록 ====================

try:
    from M1.router import router as m1_router
    app.include_router(m1_router)
    print("✅ M1 Router 등록 완료")
except Exception as e:
    print(f"⚠️ M1 Router 등록 실패: {e}")


# ==================== M2 Router 등록 ====================

try:
    from M2.router import router as m2_router
    app.include_router(m2_router)
    print("✅ M2 Router 등록 완료")
except Exception as e:
    print(f"⚠️ M2 Router 등록 실패: {e}")


# ==================== M3 Router 등록 ====================

try:
    from M3.router import router as m3_router
    app.include_router(m3_router)
    print("✅ M3 Router 등록 완료")
except Exception as e:
    print(f"⚠️ M3 Router 등록 실패: {e}")


# ==================== User Model Initialization Router 등록 ====================

try:
    from init_user_models import router as init_models_router
    app.include_router(init_models_router, prefix="/api")
    print("✅ User Model Initialization Router 등록 완료")
except Exception as e:
    print(f"⚠️ User Model Initialization Router 등록 실패: {e}")



# ==================== Pydantic Models (공통) ====================

class TrackFeatures(BaseModel):
    """트랙 오디오 특성"""
    tempo: Optional[float] = None
    energy: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None

class TrackInput(BaseModel):
    """분석할 트랙 정보"""
    trackId: int
    title: str
    artist: str
    album: Optional[str] = None
    duration: Optional[int] = None
    genre: Optional[str] = None
    audioFeatures: Optional[TrackFeatures] = None


# ==================== Root Endpoints ====================

@app.get("/")
async def root():
    """API 상태 및 엔드포인트 목록"""
    return {
        "status": "running",
        "service": "AI Music Analysis API",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "models": {
            "M1": "Audio Feature Prediction + Hybrid Recommendation",
            "M2": "TF-IDF Content-based (예정)",
            "M3": "CatBoost Collaborative Filtering (예정)"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "m1": {
                "health": "/api/m1/health",
                "analyze": "/api/m1/analyze",
                "recommend": "/api/m1/recommend/{user_id}",
                "profile": "/api/m1/user/{user_id}/profile",
                "deleted_track": "/api/m1/deleted-track",
                "retrain": "/api/m1/retrain/{user_id}"
            },
            "m2": {
                "health": "/api/m2/health",
                "predict": "/api/m2/predict",
                "recommend": "/api/m2/recommend",
                "feedback": "/api/m2/feedback",
                "train": "/api/m2/train/{user_id}",
                "retrain": "/api/m2/retrain/{user_id}"
            },
            "m3": {
                "health": "/api/m3/health",
                "analyze": "/api/m3/analyze",
                "recommend": "/api/m3/recommend",
                "train": "/api/m3/train/{user_id}",
                "models": "/api/m3/models"
            }
        }
    }


@app.get("/health")
async def health_check():
    """전체 시스템 헬스 체크"""
    health_status = {
        "status": "healthy",
        "api": True,
        "database": False,
        "models": {
            "M1": False,
            "M2": False,
            "M3": False
        }
    }
    
    # DB 연결 확인
    try:
        from database import test_connection
        health_status["database"] = test_connection()
    except:
        pass
    
    # M1 모델 확인
    try:
        model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
        health_status["models"]["M1"] = os.path.exists(model_path)
    except:
        pass
    
    # M2 모델 확인
    try:
        model_path = os.path.join(os.path.dirname(__file__), "M2", "tfidf_gbr_models.pkl")
        health_status["models"]["M2"] = os.path.exists(model_path)
    except:
        pass
    
    # M3 모델 확인
    try:
        m3_dir = os.path.join(os.path.dirname(__file__), "M3")
        cbm_files = [f for f in os.listdir(m3_dir) if f.endswith('.cbm')]
        health_status["models"]["M3"] = len(cbm_files) > 0
    except:
        pass
    
    return health_status


# ==================== Legacy Endpoints (하위 호환) ====================

@app.post("/analyze")
async def legacy_analyze(request: dict):
    """
    레거시 분석 엔드포인트 (Spring Boot 호환)
    → /api/m1/analyze로 리다이렉트
    """
    try:
        from M1.router import analyze_user, AnalyzeRequest
        from database import get_db, SessionLocal
        
        db = SessionLocal()
        try:
            req = AnalyzeRequest(userid=int(request.get("userid", 0)))
            return await analyze_user(req, db)
        finally:
            db.close()
    except Exception as e:
        return {"message": f"오류: {str(e)}"}


# ==================== EMS 데이터 분석 (공통) ====================

@app.get("/api/ems/analysis")
async def analyze_ems_data(user_id: int = Query(..., description="사용자 ID")):
    """EMS 데이터 종합 분석 - M1 프로필 조회"""
    try:
        from M1.service import M1RecommendationService
        from database import SessionLocal
        
        model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
        service = M1RecommendationService(model_path=model_path)
        
        db = SessionLocal()
        try:
            profile = service.get_user_profile(db, user_id)
            return {
                "userId": user_id,
                "profile": profile,
                "analysisDate": "2026-02-04"
            }
        finally:
            db.close()
    except Exception as e:
        return {"error": str(e)}


# ==================== 통합 추천 API (모델 선택 기반) ====================

def save_recommendations_to_gms(db, user_id: int, recommendations: list, model_name: str) -> int:
    """추천 결과를 GMS 플레이리스트로 저장"""
    from sqlalchemy import text
    from datetime import datetime
    
    try:
        # 1. GMS 플레이리스트 생성
        playlist_title = f"AI 추천 ({model_name}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # 평균 점수 계산
        avg_score = 0
        if recommendations:
            scores = []
            for r in recommendations:
                score = r.get('final_score') or r.get('probability') or r.get('recommendation_score') or 0
                if isinstance(score, (int, float)):
                    scores.append(float(score))
            if scores:
                avg_score = sum(scores) / len(scores) * 100  # 0-100 스케일
        
        insert_playlist = text("""
            INSERT INTO playlists (user_id, title, description, space_type, status_flag, source_type, ai_score, created_at, updated_at)
            VALUES (:user_id, :title, :description, 'GMS', 'PTP', 'System', :ai_score, NOW(), NOW())
        """)
        db.execute(insert_playlist, {
            "user_id": user_id,
            "title": playlist_title,
            "description": f"{model_name} 모델이 추천한 플레이리스트입니다.",
            "ai_score": avg_score
        })
        db.commit()
        
        # 생성된 플레이리스트 ID 조회
        get_playlist_id = text("SELECT LAST_INSERT_ID()")
        playlist_id = db.execute(get_playlist_id).scalar()
        
        # 2. 추천 트랙들을 플레이리스트에 추가
        for idx, rec in enumerate(recommendations):
            track_id = rec.get('track_id')
            if track_id:
                insert_track = text("""
                    INSERT INTO playlist_tracks (playlist_id, track_id, order_index, added_at)
                    VALUES (:playlist_id, :track_id, :order_index, NOW())
                    ON DUPLICATE KEY UPDATE order_index = :order_index
                """)
                db.execute(insert_track, {
                    "playlist_id": playlist_id,
                    "track_id": track_id,
                    "order_index": idx
                })
        
        db.commit()
        return playlist_id
        
    except Exception as e:
        db.rollback()
        print(f"[GMS Save Error] {e}")
        return None


class UnifiedRecommendRequest(BaseModel):
    """통합 추천 요청"""
    user_id: int
    model: str = "M1"  # M1, M2, M3
    top_k: int = 20
    ems_track_limit: int = 300  # EMS에서 분석할 곡 수 (기본값 증가)

@app.post("/api/recommend")
async def unified_recommend(request: UnifiedRecommendRequest):
    """
    통합 추천 API - 선택된 모델에 따라 라우팅
    
    - M1: Audio Feature Prediction (Ridge)
    - M2: SVM + Text Embedding (393D)
    - M3: CatBoost Collaborative Filtering
    """
    from database import SessionLocal
    
    user_id = request.user_id
    model = request.model.upper()
    ems_limit = request.ems_track_limit
    
    db = SessionLocal()
    try:
        if model == "M1":
            from M1.service import M1RecommendationService
            model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
            service = M1RecommendationService(model_path=model_path)
            
            # EMS 곡 수 설정 적용하여 추천 생성
            results = service.get_recommendations(db, user_id, ems_limit=ems_limit)
            
            if results.empty:
                return {
                    "success": False,
                    "model": "M1",
                    "message": "No recommendations found",
                    "ems_track_limit": ems_limit,
                    "recommendations": []
                }
            
            # GMS 저장
            playlist_id = service.save_gms_playlist(db, user_id, results)

            recommendations = results.head(request.top_k).fillna(0).replace([np.inf, -np.inf], 0).to_dict(orient='records')
            
            return {
                "success": True,
                "model": "M1",
                "user_id": user_id,
                "playlist_id": playlist_id,
                "ems_track_limit": ems_limit,
                "count": len(recommendations),
                "recommendations": recommendations
            }
            
        elif model == "M2":
            # M2: SVM + Text Embedding
            from M2.service import get_m2_service
            
            m2_service = get_m2_service()
            
            # EMS에서 후보 트랙 조회
            from sqlalchemy import text
            ems_query = text("""
                SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :user_id AND p.space_type = 'EMS'
                ORDER BY RAND()
                LIMIT :limit
            """)
            ems_result = db.execute(ems_query, {"user_id": user_id, "limit": ems_limit}).fetchall()
            
            if not ems_result:
                return {
                    "success": False,
                    "model": "M2",
                    "message": "EMS 데이터 없음",
                    "recommendations": []
                }
            
            # 후보 트랙 변환
            candidate_tracks = [
                {
                    'track_id': r[0],
                    'track_name': r[1],
                    'artist': r[2],
                    'album_name': r[3] or '',
                    'tags': '',
                    'duration_ms': (r[4] or 200) * 1000
                }
                for r in ems_result
            ]
            
            # M2 추천 실행
            recommendations = m2_service.get_recommendations(
                user_id=user_id,
                candidate_tracks=candidate_tracks,
                top_k=request.top_k,
                threshold=0.5
            )
            
            # GMS 플레이리스트에 저장
            playlist_id = None
            if recommendations:
                playlist_id = save_recommendations_to_gms(db, user_id, recommendations, "M2")
            
            return {
                "success": True,
                "model": "M2",
                "user_id": user_id,
                "playlist_id": playlist_id,
                "ems_track_limit": ems_limit,
                "count": len(recommendations),
                "recommendations": recommendations
            }
            
        elif model == "M3":
            # M3: CatBoost Collaborative Filtering
            from M3.service import get_m3_service
            
            m3_service = get_m3_service()
            result = m3_service.get_recommendations(db, user_id, top_k=request.top_k)
            
            if not result.get("success"):
                return result
            
            # GMS 플레이리스트에 저장
            playlist_id = None
            recommendations = result.get("recommendations", [])
            if recommendations:
                playlist_id = save_recommendations_to_gms(db, user_id, recommendations, "M3")
            
            return {
                "success": True,
                "model": "M3",
                "user_id": user_id,
                "playlist_id": playlist_id,
                "model_used": result.get("model_used"),
                "count": result.get("count", 0),
                "recommendations": recommendations
            }
            
        else:
            return {
                "success": False,
                "message": f"Unknown model: {model}. Use M1, M2, or M3"
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "model": model,
            "error": str(e)
        }
    finally:
        db.close()


@app.post("/api/analyze")
async def unified_analyze(request: dict):
    """
    통합 분석 API - 선택된 모델로 사용자 분석 및 학습
    
    Required: userid, model (M1/M2/M3)
    """
    from database import SessionLocal
    
    user_id = int(request.get("userid", 0))
    model = request.get("model", "M1").upper()
    
    if user_id == 0:
        return {"success": False, "message": "userid is required"}
    
    db = SessionLocal()
    try:
        if model == "M1":
            from M1.router import analyze_user, AnalyzeRequest
            req = AnalyzeRequest(userid=user_id)
            return await analyze_user(req, db)
            
        elif model == "M2":
            # M2 분석 (SVM 학습)
            from M2.service import get_m2_service
            from sqlalchemy import text
            
            m2_service = get_m2_service()
            
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
                    "model": "M2",
                    "user_id": user_id,
                    "message": f"PMS에 최소 5곡 이상 필요합니다 (현재: {len(positive_tracks)}곡)"
                }
            
            # EMS에서 Negative 트랙 샘플링
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
            
            # M2 모델 학습
            result = m2_service.train_user_model(
                user_id=user_id,
                positive_tracks=positive_tracks,
                negative_tracks=negative_tracks
            )
            
            return {
                "success": result.get("success", False),
                "model": "M2",
                "user_id": user_id,
                "message": result.get("message", "M2 SVM 모델 학습 완료"),
                "track_count": len(positive_tracks),
                "model_path": result.get("model_path")
            }
            
        elif model == "M3":
            # M3 분석 (CatBoost 학습 및 추천)
            from M3.service import get_m3_service
            
            m3_service = get_m3_service()
            result = m3_service.analyze_and_save_gms(db, user_id, top_k=50)
            
            return {
                "success": result.get("success", False),
                "model": "M3",
                "user_id": user_id,
                "message": result.get("message", "M3 CatBoost 모델 분석 완료"),
                "playlist_id": result.get("playlist_id"),
                "count": result.get("count", 0)
            }
            
        else:
            return {"success": False, "message": f"Unknown model: {model}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        db.close()


@app.get("/api/user/{user_id}/model")
async def get_user_model_preference(user_id: int):
    """사용자의 선택된 AI 모델 조회"""
    from database import SessionLocal
    from sqlalchemy import text
    
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT ai_model FROM user_preferences WHERE user_id = :uid"),
            {"uid": user_id}
        ).fetchone()
        
        if result:
            return {"user_id": user_id, "ai_model": result[0]}
        else:
            return {"user_id": user_id, "ai_model": "M1"}  # Default
    except Exception as e:
        return {"user_id": user_id, "ai_model": "M1", "error": str(e)}
    finally:
        db.close()


class UpdateModelRequest(BaseModel):
    """모델 변경 요청"""
    model: str = "M1"  # M1, M2, M3


@app.put("/api/user/{user_id}/model")
async def update_user_model_preference(user_id: int, request: UpdateModelRequest):
    """
    Settings에서 AI 모델 변경 및 추천 갱신

    1. user_preferences 테이블에 모델 저장
    2. 선택한 모델 재학습 (모델 파일 갱신)
    3. 선택한 모델로 GMS 추천 재생성
    """
    from database import SessionLocal
    from sqlalchemy import text
    from init_user_models import generate_gms_recommendations

    model = request.model.upper()
    if model not in ["M1", "M2", "M3"]:
        return {"success": False, "error": f"Invalid model: {model}. Must be M1, M2, or M3"}

    db = SessionLocal()
    try:
        # 0. 유저 이메일 조회 (M1 학습에 필요)
        user_result = db.execute(
            text("SELECT email FROM users WHERE user_id = :uid"),
            {"uid": user_id}
        ).fetchone()

        if not user_result:
            return {"success": False, "error": f"User not found: {user_id}"}

        email = user_result[0]

        # 1. 모델 설정 저장 (UPSERT)
        db.execute(
            text("""
                INSERT INTO user_preferences (user_id, ai_model)
                VALUES (:uid, :model)
                ON DUPLICATE KEY UPDATE ai_model = :model
            """),
            {"uid": user_id, "model": model}
        )
        db.commit()

        # 2. 선택한 모델 재학습 (모델 파일 갱신)
        retrain_result = None
        if model == "M1":
            from M1.service import MusicRecommendationService
            service = MusicRecommendationService()
            retrain_result = service.train_user_model(db, user_id, email)
            print(f"[Settings] M1 모델 재학습 완료: user_id={user_id}")
        elif model == "M2":
            from M2.service import M2RecommendationService
            m2_service = M2RecommendationService()
            retrain_result = m2_service.train_user_model(db, user_id, email)
            print(f"[Settings] M2 모델 재학습 완료: user_id={user_id}")
        elif model == "M3":
            from M3.m3_service import M3Service
            m3_service = M3Service()
            retrain_result = m3_service.train_user_model(db, user_id)
            print(f"[Settings] M3 모델 재학습 완료: user_id={user_id}")

        # 3. 선택한 모델로 GMS 추천 재생성
        gms_result = generate_gms_recommendations(user_id, db, model_name=model)

        return {
            "success": True,
            "user_id": user_id,
            "ai_model": model,
            "retrain": retrain_result,
            "gms": gms_result
        }
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        db.close()


# ==================== 장바구니 분석 API ====================

class CartAnalysisRequest(BaseModel):
    """장바구니 분석 요청"""
    userId: int
    model: str = "M1"  # M1, M2, M3

@app.post("/api/v1/evaluation/start")
async def cart_analysis(request: CartAnalysisRequest):
    """
    장바구니 분석 요청 - Spring Boot에서 호출
    
    1. 장바구니 플레이리스트 조회
    2. 해당 모델로 추천 생성
    3. 추천 결과를 GMS 플레이리스트로 저장
    """
    from database import SessionLocal
    from sqlalchemy import text
    
    user_id = request.userId
    model = request.model.upper()
    
    db = SessionLocal()
    try:
        # 1. 장바구니 플레이리스트 조회 (최근 생성된 "분석 요청" 플레이리스트)
        playlist_query = text("""
            SELECT p.playlist_id
            FROM playlists p
            WHERE p.user_id = :user_id
              AND p.title LIKE '분석 요청%'
              AND p.space_type = 'EMS'
              AND p.status_flag = 'PTP'
            ORDER BY p.created_at DESC
            LIMIT 1
        """)
        playlist_result = db.execute(playlist_query, {"user_id": user_id}).fetchone()
        
        if not playlist_result:
            return {
                "success": False,
                "message": "장바구니 플레이리스트를 찾을 수 없습니다."
            }
        
        playlist_id = playlist_result[0]
        
        # 2. 해당 모델로 추천 생성
        if model == "M1":
            from M1.service import M1RecommendationService
            model_path = os.path.join(os.path.dirname(__file__), "M1", "audio_predictor.pkl")
            service = M1RecommendationService(model_path=model_path)
            
            # EMS 플레이리스트 트랙 기반 추천
            results = service.get_recommendations(db, user_id, ems_limit=1000)
            
            if results.empty:
                return {
                    "success": False,
                    "message": "추천할 트랙이 없습니다."
                }
            
            # 3. GMS 플레이리스트에 저장
            gms_playlist_id = service.save_gms_playlist(db, user_id, results)
            
            # 플레이리스트 상태 업데이트 (GMS로 이동)
            update_status = text("""
                UPDATE playlists 
                SET status_flag = 'PRP', space_type = 'GMS'
                WHERE playlist_id = :playlist_id
            """)
            db.execute(update_status, {"playlist_id": playlist_id})
            db.commit()
            
            return {
                "success": True,
                "model": "M1",
                "userId": user_id,
                "playlistId": playlist_id,
                "gmsPlaylistId": gms_playlist_id,
                "message": "장바구니 분석 완료"
            }
            
        elif model == "M2":
            from M2.service import get_m2_service
            m2_service = get_m2_service()
            
            # EMS 플레이리스트 트랙 조회
            ems_query = text("""
                SELECT t.track_id, t.title, t.artist, t.album, t.duration, t.external_metadata
                FROM tracks t
                JOIN playlist_tracks pt ON t.track_id = pt.track_id
                JOIN playlists p ON pt.playlist_id = p.playlist_id
                WHERE p.user_id = :user_id AND p.space_type = 'EMS'
                LIMIT 1000
            """)
            ems_result = db.execute(ems_query, {"user_id": user_id}).fetchall()
            
            if not ems_result:
                return {
                    "success": False,
                    "message": "EMS 트랙이 없습니다."
                }
            
            candidate_tracks = [
                {
                    'track_id': r[0],
                    'track_name': r[1],
                    'artist': r[2],
                    'album_name': r[3] or '',
                    'tags': '',
                    'duration_ms': (r[4] or 200) * 1000
                }
                for r in ems_result
            ]
            
            recommendations = m2_service.get_recommendations(
                user_id=user_id,
                candidate_tracks=candidate_tracks,
                top_k=20,
                threshold=0.5
            )
            
            # GMS 저장
            gms_playlist_id = save_recommendations_to_gms(db, user_id, recommendations, "M2")
            
            # 플레이리스트 상태 업데이트
            update_status = text("""
                UPDATE playlists 
                SET status_flag = 'PRP', space_type = 'GMS'
                WHERE playlist_id = :playlist_id
            """)
            db.execute(update_status, {"playlist_id": playlist_id})
            db.commit()
            
            return {
                "success": True,
                "model": "M2",
                "userId": user_id,
                "playlistId": playlist_id,
                "gmsPlaylistId": gms_playlist_id,
                "message": "장바구니 분석 완료"
            }
            
        elif model == "M3":
            from M3.service import get_m3_service
            m3_service = get_m3_service()
            result = m3_service.get_recommendations(db, user_id, top_k=20)
            
            if not result.get("success"):
                return result
            
            # GMS 저장
            gms_playlist_id = None
            recommendations = result.get("recommendations", [])
            if recommendations:
                gms_playlist_id = save_recommendations_to_gms(db, user_id, recommendations, "M3")
            
            # 플레이리스트 상태 업데이트
            update_status = text("""
                UPDATE playlists 
                SET status_flag = 'PRP', space_type = 'GMS'
                WHERE playlist_id = :playlist_id
            """)
            db.execute(update_status, {"playlist_id": playlist_id})
            db.commit()
            
            return {
                "success": True,
                "model": "M3",
                "userId": user_id,
                "playlistId": playlist_id,
                "gmsPlaylistId": gms_playlist_id,
                "message": "장바구니 분석 완료"
            }
        else:
            return {
                "success": False,
                "message": f"Unknown model: {model}"
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        db.close()

# ==================== 서버 실행 ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )
