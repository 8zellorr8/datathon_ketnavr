"""Interfaces allow a different local analyzer or storage without changing policy."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import Regulation


class VideoAnalyzer(ABC):
    @abstractmethod
    def analyze(self, video_path: Path, config: dict, *, deadline: float) -> dict:
        """Analyze using the caller's absolute monotonic deadline."""


class RegulationRepository(ABC):
    @abstractmethod
    def find(self, rule_id: str) -> Regulation | None:
        """Exact identifier lookup; never infer a regulation from free text."""

    @abstractmethod
    def all(self) -> tuple[Regulation, ...]:
        """Return an immutable snapshot of the local catalogue."""
