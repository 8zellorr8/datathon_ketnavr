"""Adapter for the team's existing in-process AI service; no second HTTP queue."""
from __future__ import annotations

from pathlib import Path

from .ports import VideoAnalyzer


class LocalAIAnalyzer(VideoAnalyzer):
    def analyze(self, video_path: Path, config: dict, *, deadline: float) -> dict:
        from ai_service.forte_ai.pipeline import analyze_video

        return analyze_video(video_path, config, deadline=deadline)
