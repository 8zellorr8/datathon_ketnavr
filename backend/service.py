"""Application policy joining observed evidence to reviewed normative references."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from ai_service.forte_ai.pipeline import AnalysisError, AnalysisErrorCode
from ai_service.forte_ai.server import validate_analysis_result

from .catalogue import CATALOGUE_VERSION
from .errors import InvalidAnalysisResultError
from .ports import RegulationRepository, VideoAnalyzer

_REVIEW_NOTE = (
    "Зафиксировано визуальное событие. Применимый пункт нормативного акта "
    "не подтверждён; требуется проверка специалистом и внутреннего регламента банка."
)
_SOURCE_NOTE = (
    "Основание выбрано из локального справочника по коду события. "
    "Подтверждён источник, а не юридический вывод: специалист должен проверить "
    "применимость пункта, контекст операции и полноту видео."
)


class AnalysisService:
    def __init__(self, analyzer: VideoAnalyzer, regulations: RegulationRepository):
        self._analyzer = analyzer
        self._regulations = regulations

    def __call__(self, video_path: Path, config: dict, *, deadline: float) -> dict:
        return self.analyze(video_path, config, deadline=deadline)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise AnalysisError(
                "Application processing deadline expired",
                safe_code=AnalysisErrorCode.TIME_BUDGET_EXCEEDED,
            )

    def analyze(self, video_path: Path, config: dict, *, deadline: float) -> dict:
        self._check_deadline(deadline)
        raw = self._analyzer.analyze(Path(video_path), config, deadline=deadline)
        self._check_deadline(deadline)
        try:
            if not isinstance(raw, dict) or not isinstance(raw.get("violations"), list):
                raise ValueError("Expected analyzer result and event list")
            result = copy.deepcopy(raw)
            for event in result["violations"]:
                if not isinstance(event, dict):
                    raise ValueError("Invalid event")
                rule_id = event.get("rule_id")
                if rule_id is not None and not isinstance(rule_id, str):
                    raise ValueError("Invalid rule identifier")
                rule = self._regulations.find(rule_id) if rule_id else None
                status = event.get("verification_status")
                if status not in (None, "normative_source_verified", "needs_regulation_review"):
                    raise ValueError("Unknown verification status")
                # A visual-only result is never upgraded by a coincidental ID match.
                source = (rule.source() if rule and status != "needs_regulation_review"
                          and event.get("detection_basis") != "visual_observation" else None)
                event.update({
                    "rule_source": source,
                    "verification_status": "normative_source_verified" if source else "needs_regulation_review",
                    "detection_basis": "verified_rule" if source else "visual_observation",
                    "requires_human_review": True,
                    "regulatory_note": _SOURCE_NOTE if source else _REVIEW_NOTE,
                    "regulation_match": "matched" if source else "review_required",
                })
            validate_analysis_result(result)
            self._validate_timing(result)
            analysis = result.setdefault("analysis", {})
            analysis.update({
                "regulation_catalogue_version": CATALOGUE_VERSION,
                "regulation_lookup": "exact_rule_id",
                "coverage": "configured_checks_only",
            })
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise InvalidAnalysisResultError("Invalid result from the local analyzer") from exc
        self._check_deadline(deadline)
        return result

    @staticmethod
    def _validate_timing(result: dict) -> None:
        analysis = result.get("analysis", {})
        if not isinstance(analysis, dict):
            raise ValueError("Invalid analysis metadata")
        duration = analysis.get("duration_sec")
        if duration is None:
            return
        if (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
            raise ValueError("Invalid video duration")
        for event in result["violations"]:
            if (event["timecode_sec"] > duration + 0.05
                    or any(frame["timestamp_sec"] > duration + 0.05 for frame in event["evidence_frames"])):
                raise ValueError("Evidence timestamp is outside the video")
