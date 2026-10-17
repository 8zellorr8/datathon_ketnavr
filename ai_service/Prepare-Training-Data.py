"""Prepare human-reviewed local Qwen video SFT data; never starts training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training"
EXPECTED_COUNTS = {"development": 26, "validation": 8, "test": 8}
REVIEW_STATUSES = {"pending", "confirmed", "ambiguous", "excluded"}


class PreparationError(Exception):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PreparationError(code)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def local_path(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def training_path(value: str | Path) -> Path:
    path = local_path(value)
    require(path.is_relative_to(TRAINING), "output_must_be_inside_training_directory")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validated_dataset(args):
    manifest_path, splits_path, rules_path = map(
        local_path, (args.manifest, args.splits, args.rules)
    )
    manifest, splits, rules = map(read_json, (manifest_path, splits_path, rules_path))
    records = manifest.get("records")
    require(isinstance(records, list) and len(records) == 42, "expected_42_manifest_records")
    by_id = {}
    for record in records:
        require(isinstance(record, dict), "invalid_manifest_record")
        source_id = record.get("id")
        require(isinstance(source_id, str) and source_id not in by_id, "duplicate_or_invalid_id")
        require(record.get("label") in (0, 1), "invalid_organizer_label")
        require(isinstance(record.get("sha256"), str)
                and len(record["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in record["sha256"]), "invalid_source_sha256")
        require(isinstance(record.get("video_path"), str), "invalid_video_path")
        require(local_path(record["video_path"]).is_file(), "source_video_missing")
        require(isinstance(record.get("split_group"), str)
                and bool(record["split_group"]), "missing_split_group")
        by_id[source_id] = record
    ids = splits.get("ids")
    require(isinstance(ids, dict) and set(ids) == set(EXPECTED_COUNTS), "invalid_split_names")
    membership = {}
    group_parts, hash_parts = defaultdict(set), defaultdict(set)
    for partition, expected_count in EXPECTED_COUNTS.items():
        entries = ids[partition]
        require(isinstance(entries, list) and len(entries) == expected_count,
                "unexpected_partition_count")
        for source_id in entries:
            require(source_id in by_id and source_id not in membership,
                    "overlapping_or_unknown_split_id")
            membership[source_id] = partition
            record = by_id[source_id]
            group_parts[record["split_group"]].add(partition)
            hash_parts[record["sha256"]].add(partition)
    require(set(membership) == set(by_id), "split_does_not_cover_manifest")
    require(all(len(parts) == 1 for parts in group_parts.values()), "scene_group_crosses_splits")
    require(all(len(parts) == 1 for parts in hash_parts.values()), "duplicate_crosses_splits")
    require(isinstance(rules, list), "invalid_rules_catalog")
    enabled = {}
    for rule in rules:
        if rule.get("enabled") is True:
            rule_id = rule.get("id")
            require(isinstance(rule_id, str) and rule_id not in enabled, "invalid_rule_id")
            enabled[rule_id] = rule
    require(bool(enabled), "no_enabled_rules")
    provenance = {
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "splits_path": str(splits_path), "splits_sha256": sha256(splits_path),
        "rules_path": str(rules_path), "rules_sha256": sha256(rules_path),
    }
    return by_id, membership, enabled, provenance


def write_new_json(path: Path, value) -> None:
    # Exclusive creation: never replaces an existing review or export.
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def prepare(args) -> dict:
    by_id, membership, enabled, provenance = validated_dataset(args)
    output = training_path(args.output)
    require(not output.exists(), "review_file_already_exists")
    records = []
    for source_id in sorted(by_id, key=lambda value: int(value.split("-")[-1])):
        if membership[source_id] != "development":
            continue
        source = by_id[source_id]
        records.append({
            "id": source_id,
            "split": "development",
            "split_group": source["split_group"],
            "video_path": str(local_path(source["video_path"])),
            "sha256": source["sha256"],
            "clip_start_sec": 0.0,
            "clip_end_sec": None,
            "organizer_label": "violation" if source["label"] == 1 else "no_violation",
            "organizer_event_time_sec": source.get("event_time_sec"),
            "source_row": source.get("source_row"),
            "source_annotations": source.get("source_annotations", {}),
            "annotations_are_unverified": True,
            "review_status": "pending",
            "reviewer": "",
            "reviewed_at": None,
            "review_note": "",
            "timecodes_checked": False,
            "reviewed_answer": None,
        })
    document = {
        "schema_version": 1,
        "created_at": utc_now(),
        "purpose": "Human review draft only; no training has started.",
        "policy": "Development only. Organizer labels/timestamps and text are not approved answers. "
                  "Validation and test must never be added to this file or training exports.",
        "timestamp_origin": "Full source video, seconds from 0; no cropped clips supported.",
        "provenance": provenance,
        "allowed_event_ids": sorted(enabled),
        "records": records,
    }
    write_new_json(output, document)
    labels = Counter(record["organizer_label"] for record in records)
    return {"total_source_records": len(by_id), "development_drafts": len(records),
            "positive_drafts": labels["violation"], "negative_drafts": labels["no_violation"],
            "pending": len(records), "confirmed": 0, "validation_exported": 0,
            "test_exported": 0, "enabled_event_ids": len(enabled), "training_started": 0}


def checked_reviewed_answer(record, enabled):
    require(isinstance(record.get("reviewer"), str) and bool(record["reviewer"].strip()),
            "confirmed_review_requires_reviewer")
    reviewed_at = record.get("reviewed_at")
    require(isinstance(reviewed_at, str), "confirmed_review_requires_timestamp")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError:
        raise PreparationError("invalid_review_timestamp") from None
    require(timestamp.tzinfo is not None, "review_timestamp_requires_timezone")
    require(isinstance(record.get("review_note"), str), "invalid_review_note")
    answer = record.get("reviewed_answer")
    require(isinstance(answer, dict) and set(answer) == {"verdict", "events"},
            "reviewed_answer_requires_exact_schema")
    require(answer["verdict"] in ("violation", "no_violation"), "invalid_reviewed_verdict")
    events = answer["events"]
    require(isinstance(events, list), "events_must_be_list")
    if answer["verdict"] == "no_violation":
        require(not events, "negative_answer_must_have_empty_events")
    else:
        require(bool(events), "positive_answer_requires_event")
        require(record.get("timecodes_checked") is True, "positive_timecodes_not_manually_checked")
        require(bool(record["review_note"].strip()), "positive_review_requires_manual_review_note")
    checked_events = []
    for event in events:
        require(isinstance(event, dict) and set(event) == {"event", "timecode_sec"},
                "event_requires_exact_schema")
        require(event["event"] in enabled, "unknown_or_disabled_event_id")
        value = event["timecode_sec"]
        require(isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0, "invalid_event_timecode")
        checked_events.append({"event": event["event"], "timecode_sec": float(value)})
    # Duration is not independently decoded here. The explicit manual checkbox and
    # review note record the human's obligation to check the source duration/PTS.
    return {"verdict": answer["verdict"], "events": checked_events}


def neutral_prompt(enabled) -> str:
    catalog = [{"event": rule_id, "name": rule.get("name", rule_id),
                "visual_requirements": rule.get("visual_requirements", []),
                "counterexamples": rule.get("counterexamples", [])}
               for rule_id, rule in sorted(enabled.items())]
    return (
        "<video>\nПроанализируй видимые действия на всём видео без звука. "
        "Не устанавливай личности. Используй только наблюдаемые признаки и определения "
        "категорий ниже. Не додумывай невидимые операции или намерения. "
        "Выведи только JSON: {\"verdict\":\"violation\" или \"no_violation\","
        "\"events\":[{\"event\":\"идентификатор категории\",\"timecode_sec\":число}]}. "
        "Таймкод — секунды от начала полного видео, с 0. Для no_violation events пуст. "
        "Для violation укажи хотя бы одно подтверждённое наблюдением событие. "
        "Категории возможных признаков для проверки человеком:\n"
        + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    )


def export(args) -> dict:
    by_id, membership, enabled, provenance = validated_dataset(args)
    review_path = training_path(args.reviews)
    output = training_path(args.output)
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    require(not output.exists() and not sidecar.exists(), "export_file_already_exists")
    document = read_json(review_path)
    require(document.get("schema_version") == 1, "unsupported_review_schema")
    require(document.get("provenance") == provenance, "source_provenance_changed")
    require(document.get("allowed_event_ids") == sorted(enabled), "event_catalog_changed")
    records = document.get("records")
    require(isinstance(records, list), "invalid_review_records")
    seen, samples, metadata, statuses = set(), [], [], Counter()
    prompt = neutral_prompt(enabled)
    for record in records:
        require(isinstance(record, dict), "invalid_review_record")
        source_id = record.get("id")
        require(source_id in by_id and source_id not in seen, "unknown_or_duplicate_review_id")
        seen.add(source_id)
        require(membership[source_id] == "development" and record.get("split") == "development",
                "training_contains_withheld_record")
        source = by_id[source_id]
        path = local_path(source["video_path"])
        require(record.get("sha256") == source["sha256"]
                and local_path(record.get("video_path", "")) == path
                and record.get("split_group") == source["split_group"], "review_source_mismatch")
        require(record.get("clip_start_sec") == 0 and record.get("clip_end_sec") is None,
                "cropped_clip_export_not_supported")
        require(record.get("organizer_label") == ("violation" if source["label"] == 1 else "no_violation")
                and record.get("organizer_event_time_sec") == source.get("event_time_sec"),
                "organizer_provenance_changed")
        status = record.get("review_status")
        require(status in REVIEW_STATUSES, "invalid_review_status")
        statuses[status] += 1
        if status != "confirmed":
            continue
        answer = checked_reviewed_answer(record, enabled)
        require(sha256(path) == source["sha256"], "source_video_bytes_changed")
        samples.append({
            "video": str(path),
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False,
                                                     separators=(",", ":"), allow_nan=False)},
            ],
        })
        metadata.append({"sample_index": len(samples) - 1, "id": source_id,
                         "sha256": source["sha256"], "video_path": str(path),
                         "split_group": source["split_group"],
                         "reviewer": record["reviewer"], "reviewed_at": record["reviewed_at"],
                         "review_note": record["review_note"],
                         "timecodes_checked": record.get("timecodes_checked") is True,
                         "source_hash_checked": True,
                         "duration_checked_by_utility": False})
    expected_ids = {source_id for source_id, part in membership.items() if part == "development"}
    require(seen == expected_ids, "review_must_retain_all_development_records")
    require(bool(samples), "no_confirmed_examples_export_refused")
    provenance_sidecar = {
        "schema_version": 1, "created_at": utc_now(), "source_provenance": provenance,
        "reviews_sha256": sha256(review_path), "partition": "development",
        "timestamp_origin": "Full source video, seconds from 0.",
        "format": "Qwen video/conversations JSON; trainer compatibility not executed.",
        "holdout_limitation": "Existing validation/test annotations and baseline results were "
                              "previously reviewed; original test is not a pristine blind holdout.",
        "status_counts": dict(statuses), "samples": metadata,
    }
    # Create sidecar first; if creation of the export fails, remove only the
    # sidecar exclusively created by this invocation, preserving existing files.
    write_new_json(sidecar, provenance_sidecar)
    try:
        write_new_json(output, samples)
    except Exception:
        sidecar.unlink()
        raise
    return {"development_records": len(records), "confirmed_exported": len(samples),
            "pending_skipped": statuses["pending"], "ambiguous_skipped": statuses["ambiguous"],
            "excluded_skipped": statuses["excluded"], "validation_exported": 0,
            "test_exported": 0, "training_started": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "export"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", default="data/manifest.json")
        command.add_argument("--splits", default="data/splits.json")
        command.add_argument("--rules", default="config/rules.json")
        if name == "prepare":
            command.add_argument("--output", default="training/development.review.json")
        else:
            command.add_argument("--reviews", default="training/development.review.json")
            command.add_argument("--output", default="training/qwen-development.confirmed.json")
    args = parser.parse_args()
    try:
        report = prepare(args) if args.command == "prepare" else export(args)
    except PreparationError as error:
        print(json.dumps({"error_code": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        print('{"error_code":"file_or_schema_error"}', file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
