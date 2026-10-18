"""Immutable domain objects. A checked source is not a proven legal violation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Act:
    id: str
    title: str
    number: str
    adopted_on: str
    url: str
    history_url: str


@dataclass(frozen=True)
class Regulation:
    id: str
    name: str
    summary: str
    act: Act | None
    section: str
    source_verified: bool
    checked_at: str
    requirements: tuple[str, ...]
    counterexamples: tuple[str, ...]

    def source(self) -> dict | None:
        if not self.source_verified or self.act is None:
            return None
        return {
            "title": self.act.title, "url": self.act.url, "section": self.section,
            "verified": True, "checked_at": self.checked_at,
            "act_id": self.act.id, "act_number": self.act.number,
            "adopted_on": self.act.adopted_on, "history_url": self.act.history_url,
            "summary": self.summary,
            "applicability": list(self.requirements),
            "counterexamples": list(self.counterexamples),
        }

    def to_dict(self) -> dict:
        return {
            "rule_id": self.id, "name": self.name, "summary": self.summary,
            "rule_source": self.source(),
            "source_verified": self.source_verified,
            "checked_at": self.checked_at,
            "requirements": list(self.requirements),
            "counterexamples": list(self.counterexamples),
            "requires_human_review": True,
        }
