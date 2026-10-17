"""In-memory dictionary repository: no network calls while reviewing a video."""
from __future__ import annotations

import re
from datetime import date
from types import MappingProxyType
from urllib.parse import urlsplit

from .catalogue import ACTS, CATALOGUE_VERSION, REGULATIONS
from .errors import CatalogueError
from .models import Act, Regulation
from .ports import RegulationRepository

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text")
    return value


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("Expected text sequence")
    return tuple(_text(item) for item in value)


class DictionaryRegulationRepository(RegulationRepository):
    def __init__(self, regulations: dict | None = None, acts: dict | None = None):
        entries = REGULATIONS if regulations is None else regulations
        act_entries = ACTS if acts is None else acts
        try:
            act_objects = {}
            for identifier, data in act_entries.items():
                act = Act(id=_text(identifier), **data)
                for url in (act.url, act.history_url):
                    parsed = urlsplit(url)
                    if (parsed.scheme != "https" or parsed.netloc != "adilet.zan.kz"
                            or not parsed.path.startswith("/rus/docs/")
                            or parsed.query or parsed.fragment):
                        raise ValueError("Unsupported official source URL")
                date.fromisoformat(act.adopted_on)
                act_objects[identifier] = act
            rules = {}
            for identifier, data in entries.items():
                if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
                    raise ValueError("Invalid rule identifier")
                verified = data["source_verified"]
                if type(verified) is not bool:
                    raise ValueError("Source verification must be explicit")
                act_id = data.get("act_id")
                act = act_objects[act_id] if act_id else None
                section = data.get("section", "")
                if verified and (act is None or not _text(section)):
                    raise ValueError("Verified rule needs an act and section")
                checked_at = data["checked_at"]
                date.fromisoformat(checked_at)
                rules[identifier] = Regulation(
                    id=identifier, name=_text(data["name"]), summary=_text(data["summary"]),
                    act=act, section=section, source_verified=verified,
                    checked_at=checked_at,
                    requirements=_texts(data["requirements"]),
                    counterexamples=_texts(data["counterexamples"]),
                )
            if not rules:
                raise ValueError("Empty catalogue")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise CatalogueError("Invalid local regulation dictionary") from exc
        # Detach caller-owned dictionaries and prevent mutation after startup.
        self._rules = MappingProxyType(rules)

    def find(self, rule_id: str) -> Regulation | None:
        return self._rules.get(rule_id) if isinstance(rule_id, str) else None

    def all(self) -> tuple[Regulation, ...]:
        return tuple(self._rules.values())

    def catalogue(self, rule_id: str | None = None) -> dict | None:
        if rule_id is not None:
            rule = self.find(rule_id)
            return rule.to_dict() if rule else None
        return {
            "version": CATALOGUE_VERSION,
            "lookup": "exact_rule_id",
            "updates": "manual_review_required",
            "rules": [rule.to_dict() for rule in self.all()],
        }
