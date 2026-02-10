"""
Spotify 추천 서비스 v2 (Kuka)
===============================
Notebook 1–4 검증 결과 반영:
- KNN(FAISS) 오디오 추천 (NDCG=0.498)
- MiniLM 텍스트 임베딩 (384d, NDCG=0.403)
- MMR 다양성 제어 (λ 파라미터)
- RAG 설명 생성 (FAISS 검색 + grounding instruction)
"""

import numpy as np
import pandas as pd
import faiss
import logging
from pathlib import Path
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ========================================
# 상수
# ========================================
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


class SpotifyRecommendService:
    """Spotify 추천 엔진"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.audio_features: Optional[np.ndarray] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.faiss_audio_index: Optional[faiss.IndexFlatIP] = None
        self.faiss_text_index: Optional[faiss.IndexFlatIP] = None
        self.gemini_client = None
        self._loaded = False

    def load(self):
        """서버 시작 시 데이터 로딩"""
        if self._loaded:
            return

        logger.info("📦 Spotify 데이터 로딩 시작...")

        # 1. 곡 데이터 로드
        parquet_path = DATA_DIR / "spotify_cleaned.parquet"
        self.df = pd.read_parquet(parquet_path)
        logger.info(f"  → {len(self.df)}곡 로드")

        # 2. genres 문자열 → 리스트 변환
        import ast
        def parse_genres(g):
            # numpy.ndarray 처리
            if isinstance(g, np.ndarray):
                return g.tolist()
            if isinstance(g, list):
                return g
            if isinstance(g, str):
                try:
                    parsed = ast.literal_eval(g)
                    if isinstance(parsed, list):
                        return parsed
                except (ValueError, SyntaxError):
                    pass
                return [g]
            return []
        self.df['genres'] = self.df['genres'].apply(parse_genres)
        logger.info(f"  → genres 파싱 완료")

        # 3. 오디오 피처 정규화
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self.audio_features = scaler.fit_transform(
            self.df[AUDIO_FEATURES].values
        ).astype(np.float32)
        logger.info(f"  → 오디오 피처: {self.audio_features.shape}")

        # 3. 텍스트 임베딩 로드 또는 생성
        emb_path = MODELS_DIR / "text_embeddings.npy"
        if emb_path.exists():
            self.text_embeddings = np.load(emb_path).astype(np.float32)
            logger.info(f"  → 텍스트 임베딩 캐시 로드: {self.text_embeddings.shape}")
        else:
            self._generate_text_embeddings(emb_path)

        # 4. FAISS 인덱스 구축
        self._build_faiss_indices()

        # 5. Gemini 클라이언트 (선택)
        self._init_gemini()

        self._loaded = True
        logger.info("✅ Spotify 추천 서비스 로딩 완료")

    def _generate_text_embeddings(self, save_path: Path):
        """텍스트 임베딩 생성 (최초 1회)"""
        logger.info("  → 텍스트 임베딩 생성 중 (최초 1회, CPU 사용)...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        texts = []
        for _, row in self.df.iterrows():
            genres = row.get('genres', [])
            if isinstance(genres, list):
                genre_str = ', '.join(genres[:3])
            else:
                genre_str = str(genres)
            texts.append(f"{row['artists']} - {row['track_name']} [{genre_str}]")

        self.text_embeddings = model.encode(
            texts, show_progress_bar=True, batch_size=512
        ).astype(np.float32)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, self.text_embeddings)
        logger.info(f"  → 텍스트 임베딩 저장: {self.text_embeddings.shape}")

    def _build_faiss_indices(self):
        """FAISS 인덱스 구축 (L2 정규화 + IndexFlatIP = 코사인 유사도)"""
        # 오디오 피처 인덱스
        audio_norm = self.audio_features.copy()
        norms = np.linalg.norm(audio_norm, axis=1, keepdims=True)
        audio_norm = audio_norm / np.maximum(norms, 1e-10)
        self.faiss_audio_index = faiss.IndexFlatIP(audio_norm.shape[1])
        self.faiss_audio_index.add(audio_norm)
        self._audio_normed = audio_norm

        # 텍스트 임베딩 인덱스
        text_norm = self.text_embeddings.copy()
        norms = np.linalg.norm(text_norm, axis=1, keepdims=True)
        text_norm = text_norm / np.maximum(norms, 1e-10)
        self.faiss_text_index = faiss.IndexFlatIP(text_norm.shape[1])
        self.faiss_text_index.add(text_norm)
        self._text_normed = text_norm

        logger.info(f"  → FAISS 인덱스 구축: 오디오({self.faiss_audio_index.ntotal}), 텍스트({self.faiss_text_index.ntotal})")

    def _init_gemini(self):
        """Gemini 클라이언트 초기화"""
        try:
            import os
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCi0gePeAuBvMHLd0b-JL_reZbvN3mZunc")
            if api_key:
                from google import genai
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("  ✅ Gemini 클라이언트 초기화 완료")
            else:
                logger.warning("  ⚠️ GEMINI_API_KEY 미설정 — 설명 생성 비활성화")
        except ImportError:
            logger.warning("  ⚠️ google-genai 미설치 — 설명 생성 비활성화")

    # ========================================
    # 곡 검색
    # ========================================

    def find_tracks(self, artist: str = None, song: str = None) -> list[int]:
        """아티스트/곡명으로 인덱스 검색"""
        mask = pd.Series([True] * len(self.df))

        if artist:
            mask &= self.df['artists'].str.contains(artist, case=False, na=False)
        if song:
            mask &= self.df['track_name'].str.contains(song, case=False, na=False)

        indices = self.df[mask].index.tolist()
        return indices

    # ========================================
    # 추천 모델
    # ========================================

    def recommend_knn(self, liked_indices: list[int], k: int = 10,
                      diversity: float = 0.0) -> list[dict]:
        """
        KNN 오디오 추천 (NDCG=0.498)
        diversity=0.0: 순수 KNN
        diversity>0.0: KNN + MMR (λ = 1 - diversity)
        """
        if diversity > 0:
            return self._recommend_knn_mmr(liked_indices, k, lam=1.0 - diversity)
        return self._recommend_knn_pure(liked_indices, k)

    def _recommend_knn_pure(self, liked_indices: list[int], k: int) -> list[dict]:
        """순수 KNN (sklearn)"""
        all_indices = np.arange(len(self.df))
        candidate_mask = np.ones(len(self.df), dtype=bool)
        candidate_mask[liked_indices] = False
        candidate_indices = all_indices[candidate_mask]

        candidate_features = self.audio_features[candidate_indices]
        liked_features = self.audio_features[liked_indices]

        n_neighbors = min(10, len(candidate_indices))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        knn.fit(candidate_features)

        distances, local_indices = knn.kneighbors(liked_features)

        scores = np.zeros(len(candidate_indices))
        for i in range(len(liked_indices)):
            for j in range(n_neighbors):
                local_idx = local_indices[i, j]
                dist = max(distances[i, j], 1e-10)
                scores[local_idx] += 1.0 / dist

        top_k_local = np.argsort(scores)[::-1][:k * 3]  # 중복 대비 여유분
        top_k_global = candidate_indices[top_k_local]
        top_k_scores = scores[top_k_local]

        results = self._format_results(top_k_global, top_k_scores)
        return results[:k]

    def _recommend_knn_mmr(self, liked_indices: list[int], k: int,
                           lam: float = 0.7, n_pool: int = 200) -> list[dict]:
        """KNN + MMR: 관련성과 다양성의 균형"""
        all_indices = np.arange(len(self.df))
        candidate_mask = np.ones(len(self.df), dtype=bool)
        candidate_mask[liked_indices] = False
        candidate_indices = all_indices[candidate_mask]

        candidate_features = self.audio_features[candidate_indices]
        liked_features = self.audio_features[liked_indices]

        n_neighbors = min(10, len(candidate_indices))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        knn.fit(candidate_features)

        distances, local_indices = knn.kneighbors(liked_features)

        scores = np.zeros(len(candidate_indices))
        for i in range(len(liked_indices)):
            for j in range(n_neighbors):
                local_idx = local_indices[i, j]
                dist = max(distances[i, j], 1e-10)
                scores[local_idx] += 1.0 / dist

        # 상위 n_pool개 후보
        pool_local = np.argsort(scores)[::-1][:n_pool]
        pool_scores = scores[pool_local]
        pool_features = candidate_features[pool_local]

        # 정규화
        if pool_scores.max() > pool_scores.min():
            pool_scores_norm = (pool_scores - pool_scores.min()) / (pool_scores.max() - pool_scores.min())
        else:
            pool_scores_norm = np.ones(len(pool_scores))

        # 피처 정규화
        norms = np.linalg.norm(pool_features, axis=1, keepdims=True)
        pool_normed = pool_features / np.maximum(norms, 1e-10)

        # MMR 순차 선택
        selected = []
        remaining = list(range(len(pool_local)))

        for _ in range(k):
            if not remaining:
                break

            best_idx = -1
            best_mmr = -np.inf

            for idx in remaining:
                relevance = pool_scores_norm[idx]

                if selected:
                    selected_normed = pool_normed[selected]
                    sims_to_selected = selected_normed @ pool_normed[idx]
                    max_sim = sims_to_selected.max()
                else:
                    max_sim = 0.0

                mmr_score = lam * relevance - (1 - lam) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        top_k_global = candidate_indices[pool_local[selected]]
        top_k_scores = pool_scores[selected]

        results = self._format_results(top_k_global, top_k_scores)
        return results[:k]

    def recommend_text(self, liked_indices: list[int], k: int = 10) -> list[dict]:
        """텍스트 임베딩 추천 (FAISS, NDCG=0.403)"""
        emb = self._text_normed

        centroid = emb[liked_indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroid = centroid.reshape(1, -1).astype(np.float32)

        # liked 제외한 후보로 임시 인덱스
        all_indices = np.arange(len(self.df))
        candidate_mask = np.ones(len(self.df), dtype=bool)
        candidate_mask[liked_indices] = False
        candidate_indices = all_indices[candidate_mask]

        candidate_emb = np.ascontiguousarray(emb[candidate_indices])
        index = faiss.IndexFlatIP(candidate_emb.shape[1])
        index.add(candidate_emb)

        dists, local_indices = index.search(centroid, k * 3)  # 중복 대비 여유분
        top_k_global = candidate_indices[local_indices[0]]
        top_k_scores = dists[0]

        results = self._format_results(top_k_global, top_k_scores)
        return results[:k]

    def recommend_hybrid(self, liked_indices: list[int], k: int = 10,
                         alpha: float = 0.1) -> list[dict]:
        """하이브리드 추천 (오디오 α + 텍스트 1-α)"""
        all_indices = np.arange(len(self.df))
        candidate_mask = np.ones(len(self.df), dtype=bool)
        candidate_mask[liked_indices] = False
        candidate_indices = all_indices[candidate_mask]

        # 오디오 코사인 유사도
        audio_centroid = self._audio_normed[liked_indices].mean(axis=0)
        audio_centroid = audio_centroid / (np.linalg.norm(audio_centroid) + 1e-10)
        audio_scores = self._audio_normed[candidate_indices] @ audio_centroid

        # 텍스트 코사인 유사도
        text_centroid = self._text_normed[liked_indices].mean(axis=0)
        text_centroid = text_centroid / (np.linalg.norm(text_centroid) + 1e-10)
        text_scores = self._text_normed[candidate_indices] @ text_centroid

        # 결합
        combined = alpha * audio_scores + (1 - alpha) * text_scores
        top_k_local = np.argsort(combined)[::-1][:k * 3]  # 중복 대비 여유분
        top_k_global = candidate_indices[top_k_local]
        top_k_scores = combined[top_k_local]

        results = self._format_results(top_k_global, top_k_scores)
        return results[:k]

    # ========================================
    # RAG 설명 생성
    # ========================================

    def generate_explanation(self, liked_indices: list[int],
                             recommendations: list[dict]) -> Optional[str]:
        """RAG 기반 추천 설명 (수치 인용 + grounding instruction)"""
        if not self.gemini_client:
            return None

        # 사용자 취향 프로필
        liked_df = self.df.iloc[liked_indices]
        liked_artists = liked_df['artists'].value_counts().head(3).index.tolist()

        liked_features = self.audio_features[liked_indices]
        liked_avg = liked_features.mean(axis=0)
        liked_avg_str = ', '.join([
            f"{fn}={fv:.2f}" for fn, fv in zip(AUDIO_FEATURES, liked_avg)
        ])

        # liked songs FAISS 인덱스 (유사곡 검색용)
        liked_emb = np.ascontiguousarray(self._text_normed[liked_indices])
        liked_index = faiss.IndexFlatIP(liked_emb.shape[1])
        liked_index.add(liked_emb)

        # 각 추천곡의 RAG 컨텍스트
        context_parts = []
        for rec in recommendations:
            idx = rec['index']
            row = self.df.iloc[idx]
            genres = row.get('genres', [])
            if isinstance(genres, list):
                genre_str = ', '.join(genres[:3])
            else:
                genre_str = str(genres)

            # 오디오 피처
            track_features = self.audio_features[idx]
            feature_str = ', '.join([
                f"{fn}={fv:.2f}" for fn, fv in zip(AUDIO_FEATURES, track_features)
            ])

            # FAISS로 유사한 liked_songs 검색
            track_emb = self._text_normed[idx:idx+1]
            dists, ids = liked_index.search(track_emb, 3)

            similar_liked = []
            for lid, ldist in zip(ids[0], dists[0]):
                liked_row = self.df.iloc[liked_indices[lid]]
                similar_liked.append(
                    f"{liked_row['artists']} - {liked_row['track_name']} (유사도={ldist:.3f})"
                )

            context_parts.append(
                f"곡: {row['artists']} - {row['track_name']} [{genre_str}]\n"
                f"  오디오 특성: {feature_str}\n"
                f"  가장 유사한 좋아하는 곡: {'; '.join(similar_liked)}"
            )

        prompt = f"""당신은 음악 추천 전문가입니다. 아래 데이터를 기반으로 추천 이유를 설명하세요.

## 사용자 취향 프로필
좋아하는 아티스트: {', '.join(liked_artists)}
평균 오디오 특성: {liked_avg_str}

## 추천 곡 상세 데이터
{chr(10).join(context_parts)}

## 요구사항
1. 각 곡이 사용자 취향과 어떻게 연결되는지 1~2문장으로 설명하세요.
2. 반드시 위에 제공된 오디오 특성 수치를 직접 인용하세요 (예: "energy가 0.82로 사용자 평균 0.75와 유사하여...").
3. 위 데이터에 없는 정보는 절대 추가하지 마세요. 제공된 데이터만 사용하세요.
4. 한국어로 답변하세요. 마크다운 서식 사용하지 마세요."""

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini 호출 실패: {e}")
            return None

    # ========================================
    # 헬퍼
    # ========================================

    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> list[dict]:
        """추천 결과를 딕셔너리 리스트로 변환 (중복 제거 + 점수 정규화)"""
        # 점수 0~1 정규화
        if len(scores) > 0 and scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        elif len(scores) > 0:
            norm_scores = np.ones_like(scores)
        else:
            norm_scores = scores

        results = []
        seen = set()
        rank = 1
        for idx, score in zip(indices, norm_scores):
            row = self.df.iloc[idx]
            key = (row['track_name'], row['artists'])
            if key in seen:
                continue
            seen.add(key)

            genres = row.get('genres', [])
            if isinstance(genres, list):
                genres = genres[:3]
            else:
                genres = [str(genres)]

            results.append({
                'rank': rank,
                'index': int(idx),
                'track_name': row['track_name'],
                'artists': row['artists'],
                'genres': genres,
                'similarity': round(float(score), 4),
            })
            rank += 1
        return results

    def get_model_info(self) -> dict:
        """사용 가능한 모델 정보"""
        return {
            'models': {
                'ensemble': {
                    'description': '오디오+텍스트 앙상블 α=0.4 (NDCG=0.571, 챔피언)',
                    'supports_diversity': False,
                    'default': True,
                },
                'knn': {
                    'description': '오디오 피처 기반 KNN (NDCG=0.498)',
                    'supports_diversity': True,
                },
                'text': {
                    'description': '텍스트 임베딩 FAISS 검색 (NDCG=0.408)',
                    'supports_diversity': False,
                },
                'hybrid': {
                    'description': '오디오 + 텍스트 하이브리드 α=0.1 (NDCG=0.407)',
                    'supports_diversity': False,
                },
            },
            'total_tracks': len(self.df) if self.df is not None else 0,
            'embedding_dim': self.text_embeddings.shape[1] if self.text_embeddings is not None else 0,
            'audio_features': AUDIO_FEATURES,
            'gemini_available': self.gemini_client is not None,
        }


# 싱글톤 인스턴스
spotify_service = SpotifyRecommendService()