"""Spotify 추천 API 스키마 (Kuka)"""

from pydantic import BaseModel, Field
from typing import Optional


class TrackInfo(BaseModel):
    rank: int
    track_name: str
    artists: str
    genres: list[str] = Field(default_factory=list)
    similarity: float


class RecommendResponse(BaseModel):
    model: str
    query: str
    total_tracks: int
    recommendations: list[TrackInfo]
    explanation: Optional[str] = None
    diversity: float = 0.0


class ModelInfoResponse(BaseModel):
    models: dict
    total_tracks: int
    embedding_dim: int
    audio_features: list[str]
    gemini_available: bool