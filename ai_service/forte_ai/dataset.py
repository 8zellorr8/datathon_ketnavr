"""Import the manually reviewed ForteBank workbook without third-party packages.

The workbook contains observations and hypotheses supplied by people. They are
training/review annotations, not facts inferred by a model or legal conclusions.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile


EXPECTED_RECORD_COUNT = 42
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# Manual scene families, supplied for this particular dataset. These are not
# machine-verified scene identities or violation classes.
SCENARIO_GROUPS = (
    (1, 2, 3), (4, 5, 47, 48), (8, 20, 21), (10,),
    (11, 12, 22, 23), (13, 14), (15, 16, 17), (18, 19),
    (24, 25), (26, 27), (28, 29), (30, 31, 32, 33),
    (34, 35, 37), (38, 39), (40,), (43, 44, 45), (46,),
)
SCENARIO_BY_ID = {
    f"X-{number}": f"scenario-{index:02d}"
    for index, members in enumerate(SCENARIO_GROUPS, start=1)
    for number in members
}

KNOWN_REVIEW_NOTES = (
    {
        "ids": ["X-22", "X-27"],
        "kind": "description_label_tension",
        "message": (
            "Descriptions mention a key remaining in the safe lock in both "
            "X-22 (violation) and X-27 (no violation). This does not establish "
            "identical circumstances. Review the full context; preserve labels."
        ),
    },
    {
        "ids": ["X-2", "X-19"],
        "kind": "hypothesis_not_established",
        "message": (
            "No document presentation is described for the first customer in "
            "both clips, but their labels differ. Absence of a document in "
            "the described view is not a confirmed violation rule."
        ),
    },
)


class DatasetValidationError(ValueError):
    """Invalid input; individual problems are also written to the audit file."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Dataset validation failed:\n" + "\n".join(errors))


def normalize_video_id(value: str) -> str:
    """Normalize only an identifier, never a real filesystem path."""
    text = str(value).strip()
    if text.casefold().endswith(".mov"):
        text = text[:-4]
    match = re.fullmatch(r"[XxХх]\s*[-‐‑‒–—]\s*(\d+)", text)
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"Invalid video ID: {value!r}")
    return f"X-{int(match.group(1))}"


def _plain_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    # Reading only <t> nodes avoids accidentally importing phonetic metadata.
    return "".join(node.text or "" for node in element.iter(f"{{{NS['m']}}}t"))


def _read_workbook(path: Path) -> tuple[dict[str, str], list[tuple[int, dict[str, str]]]]:
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("m:sheets/m:sheet", NS)
        if sheet is None:
            raise ValueError("Workbook has no worksheets")
        rel_id = sheet.get(f"{{{DOC_REL_NS}}}id")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel = next(
            (item for item in relationships.findall(f"{{{REL_NS}}}Relationship")
             if item.get("Id") == rel_id), None,
        )
        if rel is None or rel.get("TargetMode") == "External":
            raise ValueError("First worksheet has no supported internal relationship")
        target = rel.get("Target", "")
        sheet_path = (target.lstrip("/") if target.startswith("/")
                      else posixpath.normpath(posixpath.join("xl", target)))
        if not sheet_path.startswith("xl/"):
            raise ValueError("Worksheet path is outside xl/")
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_strings = [
                _plain_text(item)
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
                .findall("m:si", NS)
            ]
        root = ET.fromstring(archive.read(sheet_path))
        headers: dict[str, str] = {}
        records = []
        for row in root.findall("m:sheetData/m:row", NS):
            number = int(row.get("r", "0"))
            if number < 6:
                continue
            values: dict[str, str] = {}
            for cell in row.findall("m:c", NS):
                ref = cell.get("r", "")
                match = re.fullmatch(r"([A-Z]+)\d+", ref)
                if not match or match.group(1) not in set("ABCDEFGHIJ"):
                    continue
                column = match.group(1)
                value_node = cell.find("m:v", NS)
                if cell.find("m:f", NS) is not None and value_node is None:
                    raise ValueError(f"Formula {ref} has no cached value")
                if cell.get("t") == "inlineStr":
                    value = _plain_text(cell.find("m:is", NS))
                elif cell.get("t") == "s":
                    if value_node is None:
                        raise ValueError(f"Missing shared-string index at {ref}")
                    value = shared_strings[int(value_node.text or "-1")]
                else:
                    value = "" if value_node is None else (value_node.text or "")
                values[column] = value.strip()
            if number == 6:
                headers = values
            elif any(values.values()):
                records.append((number, values))
        required = {"A": "Файл", "B": "Метка организатора", "C": "Таймкод (сек.)"}
        for column, expected in required.items():
            if headers.get(column) != expected:
                raise ValueError(
                    f"Expected {column}6={expected!r}, found {headers.get(column)!r}"
                )
        return headers, records


def _parse_label(value: str) -> int:
    labels = {"нарушение": 1, "без нарушения": 0}
    normalized = " ".join(value.casefold().split())
    if normalized not in labels:
        raise ValueError(f"Unknown organizer label: {value!r}")
    return labels[normalized]


def _parse_time(value: str) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Event time must be numeric seconds: {value!r}") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Event time must be finite and nonnegative: {value!r}")
    return seconds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    # Replace a complete sibling temporary file; readers never see partial JSON.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _assign_split_groups(records: list[dict]) -> list[dict]:
    """Merge manual families and exact duplicates into indivisible split units."""
    parent = {record["id"]: record["id"] for record in records}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        parent[find(right)] = find(left)

    by_scene: dict[str, list[str]] = {}
    by_hash: dict[str, list[dict]] = {}
    for record in records:
        by_scene.setdefault(record["source_scenario_group"], []).append(record["id"])
        by_hash.setdefault(record["sha256"], []).append(record)
    for members in by_scene.values():
        for member in members[1:]:
            union(members[0], member)
    duplicate_groups = []
    for digest, members in by_hash.items():
        if len(members) < 2:
            continue
        for member in members[1:]:
            union(members[0]["id"], member["id"])
        duplicate_groups.append({
            "sha256": digest,
            "ids": [member["id"] for member in members],
            "label_conflict": len({member["label"] for member in members}) > 1,
            "event_time_conflict": len({member["event_time_sec"] for member in members}) > 1,
            "event_times_by_id": {member["id"]: member["event_time_sec"] for member in members},
        })
    units: dict[str, list[str]] = {}
    for record in records:
        units.setdefault(find(record["id"]), []).append(record["id"])
    for record in records:
        first = min(units[find(record["id"])], key=lambda item: int(item.split("-")[1]))
        record["split_group"] = f"split-{first}"
    return duplicate_groups


def prepare_dataset(
    workbook_path: str | Path, videos_dir: str | Path, output_dir: str | Path,
) -> dict:
    """Validate 42 reviewed clips and write ``manifest.json`` plus an audit.

    Raises DatasetValidationError on invalid rows, missing or ambiguous clips,
    malformed workbooks, or unreadable videos. Labels and hypotheses are never
    changed automatically. No video frames are decoded or sent to a service.
    Returns the same manifest dictionary written to disk.
    """
    workbook = Path(workbook_path).expanduser().resolve()
    video_directory = Path(videos_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    audit = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workbook_path": str(workbook),
        "videos_dir": str(video_directory),
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "errors": errors,
        "warnings": warnings,
        "known_review_notes": list(KNOWN_REVIEW_NOTES),
        "source_scenario_groups": {
            "method": "manual, provisional scene families supplied for this dataset",
            "purpose": "Keep related scenes together when evaluating; not violation classes.",
            "groups": {
                f"scenario-{index:02d}": [f"X-{number}" for number in members]
                for index, members in enumerate(SCENARIO_GROUPS, start=1)
            },
        },
        "annotation_policy": (
            "Preserve organizer labels and user text. Role names, observations, "
            "and hypotheses are human annotations, not verified ground truth. "
            "Event timestamps are organizer annotations, not model predictions."
        ),
    }
    headers: dict[str, str] = {}
    rows: list[tuple[int, dict[str, str]]] = []
    try:
        headers, rows = _read_workbook(workbook)
    except (OSError, BadZipFile, ET.ParseError, KeyError, ValueError, IndexError) as exc:
        errors.append(f"Cannot read workbook: {exc}")
    audit["source_record_count"] = len(rows)
    if len(rows) != EXPECTED_RECORD_COUNT:
        errors.append(f"Expected {EXPECTED_RECORD_COUNT} data rows, found {len(rows)}")

    videos: dict[str, list[Path]] = {}
    if not video_directory.is_dir():
        errors.append(f"Video directory does not exist: {video_directory}")
    else:
        for path in sorted(video_directory.iterdir()):
            if not path.is_file() or path.suffix.casefold() != ".mov":
                continue
            try:
                video_id = normalize_video_id(path.stem)
            except ValueError:
                warnings.append(f"Ignoring MOV with an unrecognized filename: {path.name}")
                continue
            videos.setdefault(video_id, []).append(path.resolve())

    records = []
    seen_ids: set[str] = set()
    for row_number, values in rows:
        try:
            video_id = normalize_video_id(values.get("A", ""))
            if video_id in seen_ids:
                raise ValueError(f"Duplicate normalized ID: {video_id}")
            seen_ids.add(video_id)
            label = _parse_label(values.get("B", ""))
            event_time = _parse_time(values.get("C", ""))
            matches = videos.get(video_id, [])
            if not matches:
                raise ValueError(f"Missing video for {video_id}")
            if len(matches) > 1:
                raise ValueError(f"Ambiguous video for {video_id}: {[str(p) for p in matches]}")
            if video_id not in SCENARIO_BY_ID:
                raise ValueError(f"No manual source scenario group for {video_id}")
            path = matches[0]
            source_annotations = {
                headers.get(column, column): values.get(column, "")
                for column in "DEFGHIJ"
            }
            records.append({
                "id": video_id,
                "video_path": str(path),
                "label": label,
                "event_time_sec": event_time,
                "observation": values.get("D", ""),
                "hypothesis": values.get("J", ""),
                "notes": "\n".join(
                    f"{headers.get(column, column)}: {values[column]}"
                    for column in "EFGHI" if values.get(column)
                ),
                "sha256": _sha256(path),
                "source_row": row_number,
                "source_id": values.get("A", ""),
                "source_label": values.get("B", ""),
                "source_annotations": source_annotations,
                "source_scenario_group": SCENARIO_BY_ID[video_id],
            })
            if label == 0 and event_time is not None:
                warnings.append(f"{video_id}: no-violation label has an event timestamp; preserved")
            if label == 1 and event_time is None:
                warnings.append(f"{video_id}: violation label has no event timestamp; preserved")
        except (ValueError, OSError) as exc:
            errors.append(f"Row {row_number}: {exc}")

    records.sort(key=lambda record: int(record["id"].split("-")[1]))
    duplicate_groups = _assign_split_groups(records)
    audit["duplicate_groups"] = duplicate_groups
    audit["valid_record_count"] = len(records)
    audit["label_counts"] = {
        "violation": sum(record["label"] == 1 for record in records),
        "no_violation": sum(record["label"] == 0 for record in records),
    }
    audit["unused_video_ids"] = sorted(set(videos) - seen_ids)
    audit["split_policy"] = (
        "Keep all records sharing split_group in the same partition. split_group "
        "combines provisional manual scene families with exact SHA-256 duplicates. "
        "No train/validation split has been chosen by this importer."
    )
    for group in duplicate_groups:
        if group["label_conflict"]:
            warnings.append(f"Exact duplicate videos have conflicting labels: {group['ids']}")
        if group["event_time_conflict"]:
            warnings.append(
                "Exact duplicate videos have conflicting event timestamps: "
                f"{group['event_times_by_id']}; source annotations are preserved"
            )
    audit["status"] = "invalid" if errors else "valid"
    audit_path = destination / "dataset_audit.json"
    _write_json(audit_path, audit)
    if errors:
        raise DatasetValidationError(errors)

    manifest = {
        "schema_version": 1,
        "source_workbook": str(workbook),
        "audit_path": str(audit_path),
        "annotation_policy": audit["annotation_policy"],
        "split_policy": audit["split_policy"],
        "records": records,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest
