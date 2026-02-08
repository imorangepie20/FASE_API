"""
M2 Service - SVM + Text Embedding 기반 추천 서비스
m2.py의 핵심 로직을 서비스 클래스로 정리

주요 기능:
- 393D 피처 (384D 텍스트 임베딩 + 9D 오디오)
- SVM 기반 사용자 취향 예측
- Last.fm 태그 활용
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any

# sentence_transformers는 optional (없으면 TF-IDF fallback)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# 오디오 피처 목록
AUDIO_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
]


class AudioPredictionService:
    """오디오 피처 예측 서비스 (간소화 버전 - 규칙 기반)"""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(BASE_DIR / 'tfidf_gbr_models.pkl')
        
        self.model_path = Path(model_path) if model_path else None
        self.models = None
        self.vectorizers = None
        self._loaded = False
    
    def load_model(self) -> bool:
        """모델 로드 시도 (실패 시 규칙 기반 사용)"""
        if self._loaded:
            return True
        
        if self.model_path and self.model_path.exists():
            try:
                logger.info(f"오디오 예측 모델 로드: {self.model_path}")
                data = joblib.load(self.model_path)
                self.models = data.get('models', {})
                self.vectorizers = data.get('vectorizers', {})
                self._loaded = True
                logger.info(f"오디오 모델 로드 완료: {list(self.models.keys())}")
                return True
            except Exception as e:
                logger.warning(f"오디오 모델 로드 실패, 규칙 기반 사용: {e}")
        
        # 모델 없으면 규칙 기반 사용
        self._loaded = True
        logger.info("오디오 피처: 규칙 기반 예측 사용")
        return True
    
    def predict_single(
        self,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = "",
        duration_ms: int = 200000
    ) -> Dict[str, float]:
        """단일 트랙 오디오 피처 예측 (규칙 기반)"""
        self.load_model()
        
        # 규칙 기반 예측 (장르/태그 힌트 활용)
        text = f"{artist} {track_name} {album_name} {tags}".lower()
        
        # 기본값
        predictions = {
            'danceability': 0.5,
            'energy': 0.5,
            'speechiness': 0.1,
            'acousticness': 0.3,
            'instrumentalness': 0.1,
            'liveness': 0.2,
            'valence': 0.5,
            'tempo': 120.0,
            'loudness': -6.0
        }
        
        # 장르/키워드 기반 조정
        if any(kw in text for kw in ['edm', 'electronic', 'dance', 'club']):
            predictions['danceability'] = 0.8
            predictions['energy'] = 0.85
            predictions['acousticness'] = 0.1
        elif any(kw in text for kw in ['ballad', 'slow', 'acoustic', 'folk']):
            predictions['danceability'] = 0.3
            predictions['energy'] = 0.3
            predictions['acousticness'] = 0.8
        elif any(kw in text for kw in ['rock', 'metal', 'punk']):
            predictions['energy'] = 0.9
            predictions['loudness'] = -4.0
        elif any(kw in text for kw in ['hip hop', 'rap', 'hiphop']):
            predictions['speechiness'] = 0.3
            predictions['danceability'] = 0.7
        elif any(kw in text for kw in ['classical', 'orchestra', 'piano']):
            predictions['instrumentalness'] = 0.8
            predictions['acousticness'] = 0.9
            predictions['energy'] = 0.3
        elif any(kw in text for kw in ['jazz', 'blues']):
            predictions['acousticness'] = 0.6
            predictions['instrumentalness'] = 0.4
        
        # 곡 길이 기반 조정
        if duration_ms < 180000:  # 3분 미만
            predictions['energy'] = min(predictions['energy'] + 0.1, 1.0)
        elif duration_ms > 360000:  # 6분 이상
            predictions['instrumentalness'] = min(predictions['instrumentalness'] + 0.2, 1.0)
        
        return predictions


class EmbeddingService:
    """텍스트 임베딩 서비스 (Sentence-Transformers 또는 TF-IDF fallback)"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tfidf = None  # TF-IDF fallback
        self.use_tfidf = True  # TF-IDF 기본으로 사용
        self.embedding_dim = 500  # TF-IDF는 500D
        self._st_loading = False  # sentence-transformers 로딩 상태
        self._st_loaded = False  # sentence-transformers 로딩 완료 플래그
        
        # TF-IDF 즉시 로드
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        # sentence-transformers 비동기 백그라운드 로딩 시작
        self._load_sentence_transformers_async()
    
    def load_model(self) -> bool:
        """모델 로트 (TF-IDF 기본, sentence-transformers 비동기 백그라운드)"""
        if self.tfidf is None:
            self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            self.use_tfidf = True
        
        # TF-IDF 사용하면 바로 반환
        if self.use_tfidf:
            return True
        
        # sentence-transformers 이미 로딩된 경우
        if self.model is not None:
            return True
        
        return False
    
    def _load_sentence_transformers_async(self):
        """비동기로 sentence-transformers 로드 (백그라운드 스레드)"""
        if self._st_loading or self._st_loaded or not HAS_SENTENCE_TRANSFORMERS:
            return
        
        self._st_loading = True
        
        def load_in_background():
            try:
                logger.info(f"Sentence-Transformers 비동기 로드 시작: {self.model_name}")
                model = SentenceTransformer(self.model_name)
                self.model = model
                self._st_loaded = True
                self.use_tfidf = False
                self.embedding_dim = 384
                logger.info("Sentence-Transformers 비동기 로드 완료")
            except Exception as e:
                logger.warning(f"Sentence-Transformers 비동기 로드 실패, TF-IDF 사용: {e}")
                self._st_loaded = False
                self.use_tfidf = True
            finally:
                self._st_loading = False
        
        import threading
        thread = threading.Thread(target=load_in_background, daemon=True)
        thread.start()
        
        logger.info("TF-IDF 사용 중... Sentence-Transformers 백그라운드 로드 시작")
    
    def encode_track(self, artist: str, track_name: str, album_name: str = "", tags: str = "") -> np.ndarray:
        """단일 트랙 임베딩"""
        if not self.load_model():
            return np.zeros(self.embedding_dim)
        
        text = f"{artist} {track_name}"
        if album_name:
            text += f" {album_name}"
        if tags:
            text += f" {tags.replace('|', ' ')}"
        
        if not self.use_tfidf and self.model is not None:
            return self.model.encode([text])[0]
        else:
            # TF-IDF fallback - 단일 문서는 fit_transform 필요
            if self.tfidf is not None:
                try:
                    vec = self.tfidf.fit_transform([text]).toarray()[0]
                    return vec
                except:
                    return np.zeros(self.embedding_dim)
            return np.zeros(self.embedding_dim)
    
    def encode_tracks(self, tracks: List[Dict]) -> np.ndarray:
        """여러 트랙 임베딩"""
        if not self.load_model():
            return np.zeros((len(tracks), self.embedding_dim))
        
        texts = []
        for t in tracks:
            text = f"{t.get('artist', '')} {t.get('track_name', '')}"
            if t.get('album_name'):
                text += f" {t.get('album_name')}"
            if t.get('tags'):
                text += f" {t.get('tags', '').replace('|', ' ')}"
            texts.append(text)
        
        if not self.use_tfidf and self.model is not None:
            return self.model.encode(texts)
        else:
            # TF-IDF fallback
            if self.tfidf is not None:
                try:
                    return self.tfidf.fit_transform(texts).toarray()
                except:
                    return np.zeros((len(tracks), self.embedding_dim))
            return np.zeros((len(tracks), self.embedding_dim))


class M2RecommendationService:
    """M2 SVM 기반 추천 서비스"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.audio_service = AudioPredictionService()
        self.user_models: Dict[int, Any] = {}  # user_id -> SVM model
        self.models_dir = BASE_DIR / 'user_svm_models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._feature_dim = None  # 동적으로 설정
    
    def _create_features(
        self,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = "",
        audio_features: Optional[Dict[str, float]] = None,
        duration_ms: int = 200000
    ) -> np.ndarray:
        """피처 벡터 생성 (임베딩 + 9D 오디오)"""
        # 1. 텍스트 임베딩 (384D or 500D depending on model)
        embedding = self.embedding_service.encode_track(artist, track_name, album_name, tags)
        
        # 2. 오디오 피처 (9D)
        if audio_features is None:
            audio_features = self.audio_service.predict_single(
                artist=artist,
                track_name=track_name,
                album_name=album_name,
                tags=tags,
                duration_ms=duration_ms
            )
        
        audio_vector = np.array([
            audio_features.get(f, 0.5) for f in AUDIO_FEATURES
        ])
        
        # 3. 결합
        features = np.concatenate([embedding, audio_vector])
        
        # 피처 차원 기록
        if self._feature_dim is None:
            self._feature_dim = len(features)
            logger.info(f"M2 피처 차원: {self._feature_dim}D (임베딩: {len(embedding)}D + 오디오: 9D)")
        
        return features
    
    def predict_single(
        self,
        user_id: int,
        artist: str,
        track_name: str,
        album_name: str = "",
        tags: str = "",
        duration_ms: int = 200000
    ) -> Dict:
        """단일 트랙 SVM 예측"""
        # 사용자 모델 로드
        model = self._load_user_model(user_id)
        
        # 피처 생성
        features = self._create_features(
            artist=artist,
            track_name=track_name,
            album_name=album_name,
            tags=tags,
            duration_ms=duration_ms
        )
        
        X = features.reshape(1, -1)
        
        if model is None:
            # 모델 없으면 기본 확률 반환
            return {
                'probability': 0.5,
                'prediction': 0,
                'audio_features': self.audio_service.predict_single(
                    artist, track_name, album_name, tags, duration_ms
                )
            }
        
        try:
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            return {
                'probability': float(probability[1]) if len(probability) > 1 else 0.5,
                'prediction': int(prediction),
                'audio_features': self.audio_service.predict_single(
                    artist, track_name, album_name, tags, duration_ms
                )
            }
        except Exception as e:
            logger.error(f"SVM 예측 오류: {e}")
            return {
                'probability': 0.5,
                'prediction': 0,
                'audio_features': {}
            }
    
    def get_recommendations(
        self,
        user_id: int,
        candidate_tracks: List[Dict],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """후보 트랙 중 추천 선정"""
        results = []
        
        for track in candidate_tracks:
            result = self.predict_single(
                user_id=user_id,
                artist=track.get('artist', ''),
                track_name=track.get('track_name', ''),
                album_name=track.get('album_name', ''),
                tags=track.get('tags', ''),
                duration_ms=track.get('duration_ms', 200000)
            )
            
            result['artist'] = track.get('artist', '')
            result['track_name'] = track.get('track_name', '')
            result['track_id'] = track.get('track_id')
            results.append(result)
        
        # threshold 이상만 필터링
        filtered = [r for r in results if r['probability'] >= threshold]
        
        # 확률 높은 순 정렬
        sorted_results = sorted(filtered, key=lambda x: x['probability'], reverse=True)
        
        return sorted_results[:top_k]
    
    def _load_user_model(self, user_id: int) -> Optional[Any]:
        """사용자 SVM 모델 로드"""
        if user_id in self.user_models:
            return self.user_models[user_id]
        
        model_path = self.models_dir / f"user_{user_id}_svm.pkl"
        
        if not model_path.exists():
            logger.info(f"사용자 {user_id} SVM 모델 없음")
            return None
        
        try:
            model = joblib.load(model_path)
            self.user_models[user_id] = model
            logger.info(f"사용자 {user_id} SVM 모델 로드 완료")
            return model
        except Exception as e:
            logger.error(f"사용자 {user_id} SVM 모델 로드 실패: {e}")
            return None
    
    def train_user_model(
        self,
        user_id: int,
        positive_tracks: List[Dict],
        negative_tracks: List[Dict]
    ) -> Dict:
        """사용자 SVM 모델 학습"""
        if len(positive_tracks) < 5:
            return {
                "success": False,
                "message": "최소 5곡 이상의 positive 샘플이 필요합니다"
            }
        
        try:
            # 피처 생성
            X_list = []
            y_list = []
            
            for track in positive_tracks:
                features = self._create_features(
                    artist=track.get('artist', ''),
                    track_name=track.get('track_name', ''),
                    album_name=track.get('album_name', ''),
                    tags=track.get('tags', ''),
                    duration_ms=track.get('duration_ms', 200000)
                )
                X_list.append(features)
                y_list.append(1)
            
            for track in negative_tracks:
                features = self._create_features(
                    artist=track.get('artist', ''),
                    track_name=track.get('track_name', ''),
                    album_name=track.get('album_name', ''),
                    tags=track.get('tags', ''),
                    duration_ms=track.get('duration_ms', 200000)
                )
                X_list.append(features)
                y_list.append(0)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # SVM 학습
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42))
            ])
            
            pipeline.fit(X, y)
            
            # 모델 저장
            model_path = self.models_dir / f"user_{user_id}_svm.pkl"
            joblib.dump(pipeline, model_path)
            
            # 캐시 업데이트
            self.user_models[user_id] = pipeline
            
            logger.info(f"사용자 {user_id} SVM 모델 학습 완료: {model_path}")
            
            return {
                "success": True,
                "message": f"사용자 {user_id} 모델 학습 완료",
                "user_id": user_id,
                "model_path": str(model_path),
                "positive_count": len(positive_tracks),
                "negative_count": len(negative_tracks)
            }
            
        except Exception as e:
            logger.error(f"SVM 모델 학습 오류: {e}")
            return {
                "success": False,
                "message": f"학습 오류: {str(e)}"
            }


# 싱글톤 인스턴스
_m2_service: Optional[M2RecommendationService] = None

def get_m2_service() -> M2RecommendationService:
    """M2 서비스 싱글톤"""
    global _m2_service
    if _m2_service is None:
        _m2_service = M2RecommendationService()
    return _m2_service
