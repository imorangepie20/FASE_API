"""
Search-based Quality Enhancement Module
========================================

Enhances track evaluation with web search-based qualitative information:
- Artist style and influences
- Track mood and atmosphere
- Critical reception
- Playlist context analysis

Author: AI Assistant
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import re
from collections import Counter


class SearchBasedEnhancer:
    """
    Enhances track data with search-based qualitative information
    """
    
    def __init__(self):
        self.cache = {}
        self.mood_keywords = {
            'energetic': ['energetic', 'upbeat', 'lively', 'dynamic', 'powerful'],
            'calm': ['calm', 'peaceful', 'relaxing', 'soothing', 'mellow'],
            'happy': ['happy', 'joyful', 'cheerful', 'uplifting', 'positive'],
            'sad': ['sad', 'melancholic', 'emotional', 'dark', 'somber'],
            'aggressive': ['aggressive', 'intense', 'heavy', 'hard', 'fierce'],
            'romantic': ['romantic', 'love', 'intimate', 'sensual', 'tender']
        }
        
    def analyze_playlist_context(self, playlist_name: str) -> Dict[str, float]:
        """
        Infer intended mood/purpose from playlist name
        """
        if not playlist_name or pd.isna(playlist_name):
            return {'energy': 0.5, 'valence': 0.5, 'focus': 'general'}
        
        name_lower = playlist_name.lower()
        
        # Context patterns
        contexts = {
            'workout': {'energy': 0.9, 'valence': 0.7, 'tempo': 'fast'},
            'gym': {'energy': 0.9, 'valence': 0.7, 'tempo': 'fast'},
            'study': {'energy': 0.3, 'valence': 0.5, 'instrumentalness': 0.8},
            'focus': {'energy': 0.3, 'valence': 0.5, 'instrumentalness': 0.7},
            'chill': {'energy': 0.3, 'valence': 0.6, 'acousticness': 0.6},
            'relax': {'energy': 0.2, 'valence': 0.6, 'acousticness': 0.7},
            'sleep': {'energy': 0.1, 'valence': 0.5, 'acousticness': 0.8},
            'party': {'energy': 0.9, 'valence': 0.8, 'danceability': 0.9},
            'sad': {'energy': 0.3, 'valence': 0.2, 'acousticness': 0.6},
            'happy': {'energy': 0.7, 'valence': 0.9},
            'morning': {'energy': 0.6, 'valence': 0.7},
            'night': {'energy': 0.4, 'valence': 0.5},
            'driving': {'energy': 0.7, 'valence': 0.6},
            'running': {'energy': 0.9, 'valence': 0.7, 'tempo': 'fast'},
            'romantic': {'energy': 0.4, 'valence': 0.6, 'acousticness': 0.5},
            'dinner': {'energy': 0.3, 'valence': 0.6, 'acousticness': 0.6},
            'meditation': {'energy': 0.1, 'valence': 0.5, 'instrumentalness': 0.9}
        }
        
        # Match patterns
        for keyword, attributes in contexts.items():
            if keyword in name_lower:
                return attributes
        
        # Default
        return {'energy': 0.5, 'valence': 0.5, 'focus': 'general'}
    
    def extract_mood_from_text(self, text: str) -> Dict[str, float]:
        """
        Extract mood scores from descriptive text
        """
        if not text or pd.isna(text):
            return {}
        
        text_lower = text.lower()
        mood_scores = {}
        
        for mood, keywords in self.mood_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                mood_scores[mood] = min(score / len(keywords), 1.0)
        
        return mood_scores
    
    def infer_from_genre(self, genre: str) -> Dict[str, float]:
        """
        Infer typical audio characteristics from genre
        """
        if not genre or pd.isna(genre):
            return {}
        
        genre_lower = genre.lower()
        
        # Genre-based heuristics
        genre_profiles = {
            'edm': {'energy': 0.9, 'danceability': 0.85, 'valence': 0.7, 'acousticness': 0.1},
            'electronic': {'energy': 0.75, 'danceability': 0.8, 'acousticness': 0.15},
            'hip hop': {'energy': 0.7, 'danceability': 0.75, 'speechiness': 0.3},
            'rap': {'energy': 0.7, 'danceability': 0.7, 'speechiness': 0.4},
            'rock': {'energy': 0.8, 'loudness': 0.7, 'acousticness': 0.2},
            'metal': {'energy': 0.95, 'loudness': 0.9, 'acousticness': 0.1},
            'pop': {'energy': 0.65, 'danceability': 0.7, 'valence': 0.65},
            'classical': {'energy': 0.4, 'acousticness': 0.9, 'instrumentalness': 0.85},
            'jazz': {'energy': 0.5, 'acousticness': 0.7, 'instrumentalness': 0.6},
            'blues': {'energy': 0.45, 'acousticness': 0.65, 'valence': 0.4},
            'country': {'energy': 0.55, 'acousticness': 0.6, 'valence': 0.6},
            'folk': {'energy': 0.4, 'acousticness': 0.8, 'valence': 0.55},
            'acoustic': {'energy': 0.4, 'acousticness': 0.9, 'valence': 0.6},
            'indie': {'energy': 0.6, 'acousticness': 0.5, 'valence': 0.6},
            'indie-pop': {'energy': 0.6, 'acousticness': 0.5, 'valence': 0.6},
            'r&b': {'energy': 0.6, 'danceability': 0.7, 'valence': 0.6},
            'soul': {'energy': 0.55, 'valence': 0.65, 'acousticness': 0.5},
            'reggae': {'energy': 0.65, 'danceability': 0.75, 'valence': 0.7},
            'latin': {'energy': 0.75, 'danceability': 0.85, 'valence': 0.75},
            'ambient': {'energy': 0.2, 'instrumentalness': 0.8, 'acousticness': 0.4},
            'lofi': {'energy': 0.3, 'valence': 0.5, 'acousticness': 0.4}
        }
        
        # Find matching genre
        for genre_key, profile in genre_profiles.items():
            if genre_key in genre_lower:
                return profile
        
        return {}
    
    def analyze_duration_context(self, duration_ms: float) -> Dict[str, str]:
        """
        Infer track purpose from duration
        """
        duration_sec = duration_ms / 1000
        
        if duration_sec < 120:
            return {'type': 'intro/interlude', 'purpose': 'transition'}
        elif duration_sec < 180:
            return {'type': 'standard', 'purpose': 'radio-friendly'}
        elif duration_sec < 300:
            return {'type': 'extended', 'purpose': 'album track'}
        else:
            return {'type': 'epic', 'purpose': 'artistic/progressive'}
    
    def enhance_track_batch(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance a batch of tracks with inferred qualitative data
        """
        enhanced = tracks_df.copy()
        
        # Playlist context
        if 'playlist_name' in enhanced.columns:
            contexts = enhanced['playlist_name'].apply(self.analyze_playlist_context)
            for key in ['energy', 'valence', 'danceability', 'acousticness', 'instrumentalness']:
                enhanced[f'context_{key}'] = contexts.apply(lambda x: x.get(key, 0.5))
        
        # Genre-based inference
        if 'track_genre' in enhanced.columns:
            genre_profiles = enhanced['track_genre'].apply(self.infer_from_genre)
            for key in ['energy', 'valence', 'danceability', 'acousticness']:
                enhanced[f'genre_{key}'] = genre_profiles.apply(lambda x: x.get(key, 0.5))
        
        # Duration context
        if 'duration_ms' in enhanced.columns:
            duration_context = enhanced['duration_ms'].apply(self.analyze_duration_context)
            enhanced['duration_type'] = duration_context.apply(lambda x: x.get('type', 'standard'))
            enhanced['duration_purpose'] = duration_context.apply(lambda x: x.get('purpose', 'general'))
        
        # Combined qualitative score
        qual_features = [col for col in enhanced.columns if col.startswith('context_') or col.startswith('genre_')]
        if qual_features:
            enhanced['qualitative_confidence'] = enhanced[qual_features].notna().sum(axis=1) / len(qual_features)
        
        return enhanced


class IntegratedRecommender:
    """
    Integrates search enhancement with audio feature prediction
    """
    
    def __init__(self, audio_predictor, user_profile, search_enhancer, preference_classifier=None):
        self.audio_predictor = audio_predictor
        self.user_profile = user_profile
        self.search_enhancer = search_enhancer
        self.preference_classifier = preference_classifier
    
    def recommend_with_search(self, test_df: pd.DataFrame, 
                             weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate recommendations with search enhancement
        
        점수 계산 방식 (v2 - 더 엄격한 평가):
        - audio_score: 사용자 프로필과의 유사도 (0~1, 절대적 코사인 유사도)
        - qualitative_score: 장르 기반 품질 점수 (0~1)
        - genre_score: 장르 일치도 (0~1, 정확한 매칭 필요)
        - artist_score: 아티스트 친숙도 (0 또는 1)
        - preference_score: 학습된 선호도 분류기 점수 (0~1)
        """
        if weights is None:
            weights = {
                'predicted_audio': 0.40,  # 오디오 특성 유사도 (핵심)
                'qualitative': 0.15,       # 품질 점수 (보조)
                'genre': 0.25,             # 장르 일치도 (중요)
                'artist': 0.10,            # 아티스트 친숙도 (보너스)
                'preference': 0.10 if self.preference_classifier and self.preference_classifier.is_trained else 0.0
            }
        
        print("\n INTEGRATED RECOMMENDATION PIPELINE (v2 - Strict Scoring)")
        print("=" * 60)
        
        # Step 1: Predict audio features
        print("\n1  Predicting audio features...")
        with_predictions = self.audio_predictor.predict(test_df)
        
        # Step 2: Add search-based enhancement
        print("\n2  Adding search-based qualitative analysis...")
        enhanced = self.search_enhancer.enhance_track_batch(with_predictions)
        
        # Step 3: Calculate scores
        print("\n3  Calculating recommendation scores (strict mode)...")
        
        # ========== Audio Score (코사인 유사도 기반, 정규화 없이 절대값 사용) ==========
        pred_audio_features = [col for col in enhanced.columns if col.startswith('predicted_')]
        if pred_audio_features:
            user_pref_audio = np.array([
                self.user_profile.feature_stats.get(feat, {}).get('mean', 0.5)
                for feat in pred_audio_features
            ])
            user_pref_std = np.array([
                self.user_profile.feature_stats.get(feat, {}).get('std', 0.2)
                for feat in pred_audio_features
            ])
            
            test_audio = enhanced[pred_audio_features].fillna(0.5).values
            
            # 유클리드 거리 기반 유사도 (차이가 작을수록 높은 점수)
            # 각 특성별 차이를 표준편차로 정규화하여 계산
            audio_scores = []
            for i in range(len(test_audio)):
                track_features = test_audio[i]
                # 각 특성별 z-score 차이 계산
                z_diffs = np.abs(track_features - user_pref_audio) / (user_pref_std + 0.1)
                # 평균 z-score 차이를 점수로 변환 (차이가 클수록 점수 낮음)
                avg_z_diff = np.mean(z_diffs)
                # z-score 2 이상이면 0점, 0이면 1점
                score = max(0, 1 - (avg_z_diff / 2))
                audio_scores.append(score)
            audio_scores = np.array(audio_scores)
        else:
            audio_scores = np.ones(len(enhanced)) * 0.5
        
        # ========== Qualitative Score (장르 프로필 기반) ==========
        qual_features = [col for col in enhanced.columns if col.startswith('genre_')]
        if qual_features and len(qual_features) > 0:
            # 장르 기반 특성과 사용자 선호 특성 비교
            qual_scores = []
            for idx, row in enhanced.iterrows():
                qual_score = 0.5  # 기본값
                matches = 0
                for feat in qual_features:
                    feat_name = feat.replace('genre_', 'predicted_')
                    if feat_name in self.user_profile.feature_stats:
                        user_val = self.user_profile.feature_stats[feat_name].get('mean', 0.5)
                        genre_val = row.get(feat, 0.5)
                        # 차이가 0.2 이내면 일치로 판단
                        if abs(genre_val - user_val) < 0.25:
                            qual_score += 0.1
                            matches += 1
                qual_scores.append(min(qual_score, 1.0))
            qual_scores = np.array(qual_scores)
        else:
            qual_scores = np.ones(len(enhanced)) * 0.5
        
        # ========== Genre Score (정확한 장르 매칭) ==========
        if 'track_genre' in enhanced.columns:
            genre_scores = []
            for genre in enhanced['track_genre'].fillna('unknown'):
                genre_list = [g.strip().lower() for g in str(genre).split(',')]
                
                # unknown 장르는 중립 점수 0.3 (약간 페널티)
                if genre_list == ['unknown'] or genre_list == ['']:
                    genre_scores.append(0.3)
                    continue
                
                # 사용자 장르 분포에서 해당 장르의 비율 합산
                score = 0
                matched = False
                for g in genre_list:
                    if g in ['unknown', '']:
                        continue
                    # 완전 일치만 카운트 (부분 일치 X)
                    for user_genre, user_score in self.user_profile.genre_distribution.items():
                        if g == user_genre.lower() and g != 'unknown':
                            score += user_score
                            matched = True
                
                # 매칭된 장르가 없으면 0.2
                if not matched:
                    genre_scores.append(0.2)
                else:
                    # 최대 1점으로 제한
                    genre_scores.append(min(score, 1.0))
            genre_scores = np.array(genre_scores)
        else:
            genre_scores = np.zeros(len(enhanced))
        
        # ========== Artist Score (아티스트 친숙도 - 보너스) ==========
        if 'artists' in enhanced.columns:
            artist_scores = enhanced['artists'].apply(
                lambda x: 1.0 if str(x) in self.user_profile.artist_list else 0.0
            ).values
        else:
            artist_scores = np.zeros(len(enhanced))
        
        # ========== Preference Classifier Score ==========
        if self.preference_classifier and self.preference_classifier.is_trained:
            classifier_scores = self.preference_classifier.predict_score(enhanced, pred_audio_features)
        else:
            classifier_scores = np.zeros(len(enhanced))
        
        # ========== Final Score 계산 (가중 합계, 정규화 없음) ==========
        # preference가 없으면 다른 가중치 재분배
        total_weight = sum(weights.values())
        
        final_score = (
            (weights['predicted_audio'] / total_weight) * audio_scores +
            (weights['qualitative'] / total_weight) * qual_scores +
            (weights['genre'] / total_weight) * genre_scores +
            (weights['artist'] / total_weight) * artist_scores +
            (weights.get('preference', 0.0) / total_weight) * classifier_scores
        )
        
        # NaN/Inf 값 처리
        audio_scores = np.nan_to_num(audio_scores, nan=0.5, posinf=1.0, neginf=0.0)
        qual_scores = np.nan_to_num(qual_scores, nan=0.5, posinf=1.0, neginf=0.0)
        genre_scores = np.nan_to_num(genre_scores, nan=0.0, posinf=1.0, neginf=0.0)
        artist_scores = np.nan_to_num(artist_scores, nan=0.0, posinf=1.0, neginf=0.0)
        classifier_scores = np.nan_to_num(classifier_scores, nan=0.0, posinf=1.0, neginf=0.0)
        final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Add to dataframe
        enhanced['audio_score'] = np.round(audio_scores, 4)
        enhanced['qualitative_score'] = np.round(qual_scores, 4)
        enhanced['genre_score'] = np.round(genre_scores, 4)
        enhanced['artist_score'] = np.round(artist_scores, 4)
        enhanced['preference_score'] = np.round(classifier_scores, 4)
        enhanced['final_score'] = np.round(final_score, 4)
        enhanced['rank'] = enhanced['final_score'].rank(ascending=False, method='first').astype(int)
        
        # Sort
        result = enhanced.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        # 통계 출력
        print("\n Recommendation completed!")
        print(f"    Total tracks evaluated: {len(result)}")
        print(f"    Score range: {final_score.min():.3f} - {final_score.max():.3f}")
        print(f"    Score distribution:")
        print(f"      - 0.8+ (Excellent): {(final_score >= 0.8).sum()} tracks")
        print(f"      - 0.7-0.8 (Good): {((final_score >= 0.7) & (final_score < 0.8)).sum()} tracks")
        print(f"      - 0.6-0.7 (Fair): {((final_score >= 0.6) & (final_score < 0.7)).sum()} tracks")
        print(f"      - 0.5-0.6 (Low): {((final_score >= 0.5) & (final_score < 0.6)).sum()} tracks")
        print(f"      - <0.5 (Poor): {(final_score < 0.5).sum()} tracks")
        
        return result
