"""Grouped evaluation; annotations never enter the video inference function."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from datetime import datetime, timezone

from .pipeline import analyze_video
from .server import _validate_analysis_result


KNOWN_PRIOR_INFERENCE_IDS = ('X-26', 'X-46')


def _json_digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def build_run_fingerprint(config: dict, *, run_metadata: dict | None = None,
                          code_root: str | Path | None = None) -> tuple[str, dict]:
    """Record local code/prompts/rules and a caller-supplied model digest.

    No runtime/model request is made. expected_digest is a declaration, not a
    claim that the evaluator independently verified the installed model.
    """
    package = Path(code_root) if code_root is not None else Path(__file__).resolve().parent
    code_hashes = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob('*.py'))
    }
    rules_path = Path(config['rules_path']).resolve() if config.get('rules_path') else None
    rules_hash = hashlib.sha256(rules_path.read_bytes()).hexdigest() if rules_path is not None else None
    supplied = dict(run_metadata or {})
    declared_digest = supplied.get('model_digest', config.get('expected_digest'))
    if declared_digest is not None and (not isinstance(declared_digest, str) or not declared_digest):
        raise ValueError('A declared model digest must be a nonempty string.')
    components = {
        'fingerprint_version': 2,
        'config_sha256': _json_digest(config),
        'code_sha256': code_hashes,
        'prompt_provenance': 'Prompts embedded in local Python sources are covered by code_sha256.',
        'rules_content_sha256': rules_hash,
        'model': config.get('model'),
        'declared_model_digest': declared_digest,
        'model_digest_provenance': 'caller_run_metadata' if 'model_digest' in supplied else
                                   'config_expected_digest' if declared_digest else 'not_recorded',
        'model_digest_verified_by_evaluator': False,
        'caller_run_metadata': supplied,
    }
    # Optional observational checks are separate from verified normative rules.
    # Hash their bytes, not just the configured path, so an in-place edit
    # cannot reuse an evaluation fingerprint. Legacy runs without this
    # catalogue retain the same component layout and fingerprint version.
    visual_checks_path = config.get('visual_checks_path')
    if visual_checks_path:
        components['visual_checks_content_sha256'] = hashlib.sha256(
            Path(visual_checks_path).resolve().read_bytes()
        ).hexdigest()
    return _json_digest(components), components


def make_splits(manifest: dict, seed: int = 42) -> dict:
    groups = defaultdict(list)
    for record in manifest['records']:
        groups[record['split_group']].append(record)
    keys = sorted(groups)
    if len(keys) < 5:
        raise ValueError('At least five independent groups are required')
    rng = random.Random(seed)
    n_val = max(1, round(len(keys) * .2))
    n_test = max(1, round(len(keys) * .2))
    best = None
    for _ in range(500):
        shuffled = keys[:]
        rng.shuffle(shuffled)
        assignment = {
            'validation': shuffled[:n_val], 'test': shuffled[n_val:n_val + n_test],
            'development': shuffled[n_val + n_test:],
        }
        partitions = {name: [r for key in part for r in groups[key]] for name, part in assignment.items()}
        if any({r['label'] for r in part} != {0, 1} for part in partitions.values()):
            continue
        targets = {'development': .6, 'validation': .2, 'test': .2}
        total = len(manifest['records'])
        score = (2 * sum(abs(len(part) / total - targets[name]) for name, part in partitions.items())
                 + sum(abs(sum(r['label'] for r in part) / len(part) - .5) for part in partitions.values()))
        if best is None or score < best[0]:
            best = (score, assignment, partitions)
    if best is None:
        raise ValueError('Could not form three grouped partitions containing both labels')
    _, assignment, partitions = best
    return {
        'seed': seed,
        'policy': 'Provisional human scenario groups and exact duplicates are kept together. '
                  'All annotations have already been reviewed; this is a future tuning split, not a pristine blind test.',
        'groups': assignment,
        'ids': {name: [r['id'] for r in rows] for name, rows in partitions.items()},
        'counts': {name: {'total': len(rows), 'violation': sum(r['label'] for r in rows),
                           'no_violation': sum(1-r['label'] for r in rows)} for name, rows in partitions.items()},
    }


def _checked_prediction(record: dict, result) -> tuple[dict | None, str | None]:
    if result is None:
        return None, 'missing_prediction'
    if not isinstance(result, dict):
        return None, 'malformed_prediction'
    if result.get('status') != 'done':
        return None, 'analysis_failed' if result.get('status') == 'failed' else 'unfinished_prediction'
    try:
        _validate_analysis_result(result)
        # Reject non-JSON extras and non-finite metadata before saving a result.
        json.dumps(result, allow_nan=False)
        analysis = result.get('analysis', {})
        if not isinstance(analysis, dict):
            raise ValueError('analysis must be an object')
        for field in ('elapsed_sec', 'duration_sec'):
            value = analysis.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid analysis time metadata')
        duration = analysis.get('duration_sec', record.get('duration_sec'))
        if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0):
            raise ValueError('Invalid video duration')
        for event in result['violations']:
            if not event['description'].strip():
                raise ValueError('Empty event description')
            if duration is not None and (event['timecode_sec'] > duration or any(
                    frame['timestamp_sec'] > duration for frame in event['evidence_frames'])):
                raise ValueError('Event evidence falls outside the video')
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        return None, 'malformed_prediction'
    return result, None


def _content_groups(records: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        # Legacy/synthetic rows lacking SHA remain separate, not falsely merged.
        groups[record.get('sha256') or 'id:' + record['id']].append(record)
    return dict(groups)


def _acceptable_times(members: list[dict]) -> list[float]:
    return sorted({float(record['event_time_sec']) for record in members
                   if record['label'] and record.get('event_time_sec') is not None})


def _summarize_rows(records: list[dict], results: dict[str, dict], *,
                    localization_targets: dict[str, list[float]] | None = None) -> dict:
    counts = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}
    failures, localization, latencies = [], [], []
    failure_reasons = {}
    event_count = 0
    for record in records:
        result, failure = _checked_prediction(record, results.get(record['id']))
        if failure:
            failures.append(record['id'])
            failure_reasons[record['id']] = failure
            continue
        predicted = result['verdict'] == 'violation'
        key = ('true_positive' if predicted else 'false_negative') if record['label'] else ('false_positive' if predicted else 'true_negative')
        counts[key] += 1
        event_count += len(result['violations'])
        elapsed = result.get('analysis', {}).get('elapsed_sec')
        if isinstance(elapsed, (int, float)):
            latencies.append(float(elapsed))
        targets = (localization_targets.get(record['id'], []) if localization_targets is not None else
                   [record['event_time_sec']] if record.get('event_time_sec') is not None else [])
        if record['label'] and predicted and targets:
            localization.append(min(abs(float(v['timecode_sec']) - target)
                                    for v in result['violations'] for target in targets))
    completed = sum(counts.values())
    tp, fp, fn = counts['true_positive'], counts['false_positive'], counts['false_negative']
    return {
        'selected_videos': len(records), 'completed': completed, 'failed_or_unprocessed': len(failures),
        'failed_ids': failures, 'confusion_matrix_completed_only': counts,
        'failure_reasons_by_id': failure_reasons,
        'prediction_event_count_completed_only': event_count,
        'accuracy_completed_only': (tp + counts['true_negative']) / completed if completed else None,
        'precision_completed_only': tp / (tp + fp) if tp + fp else None,
        'recall_completed_only': tp / (tp + fn) if tp + fn else None,
        'correct_fraction_including_failures': (tp + counts['true_negative']) / len(records) if records else None,
        'median_localization_error_sec': statistics.median(localization) if localization else None,
        'localization_matched_positive_videos': len(localization),
        'mean_latency_sec_completed_only': statistics.mean(latencies) if latencies else None,
        'max_latency_sec_completed_only': max(latencies) if latencies else None,
        'interpretation': 'Prototype benchmark. Types in the workbook are hypotheses; no type accuracy is claimed. '
                          'One supplied timestamp per positive clip cannot exhaustively score multiple events. '
                          'Nearest-event localization may improve through overprediction; event counts are reported. '
                          'All annotations were previously seen. X-26 and X-46 were used in an earlier prototype: this is not a blind holdout.',
    }


def summarize(records: list[dict], results: dict[str, dict]) -> dict:
    """Preserve per-ID metrics and add unique-content/ambiguity-aware views."""
    report = _summarize_rows(records, results)
    groups = _content_groups(records)
    targets, conflicts, representatives, unique_results, group_details = {}, [], [], {}, []
    for digest, members in groups.items():
        members = sorted(members, key=lambda item: (int(item['id'].split('-')[-1]), item['id']))
        canonical = members[0]
        acceptable = _acceptable_times(members)
        for member in members:
            targets[member['id']] = acceptable
        if len(acceptable) > 1:
            conflicts.append({
                'sha256': canonical.get('sha256'), 'ids': [member['id'] for member in members],
                'original_event_times_by_id': {member['id']: member.get('event_time_sec') for member in members},
                'acceptable_gt_timestamps_sec': acceptable,
            })
        label_conflict = len({member['label'] for member in members}) > 1
        representatives.append(canonical)
        unique_results[canonical['id']] = (
            {'status': 'failed', 'error': {'type': 'GroundTruthLabelConflict'}} if label_conflict else
            results.get(canonical['id'])
        )
        outcomes = {}
        for member in members:
            checked, failure = _checked_prediction(member, results.get(member['id']))
            outcomes[member['id']] = checked['verdict'] if checked is not None else failure
        group_details.append({
            'canonical_id': canonical['id'], 'sha256': canonical.get('sha256'),
            'aliases': [member['id'] for member in members[1:]],
            'ground_truth_label_conflict': label_conflict,
            'acceptable_gt_timestamps_sec': acceptable,
            'prediction_outcomes_by_id': outcomes,
            'prediction_disagreement': len(set(outcomes.values())) > 1,
        })
    ambiguity_report = _summarize_rows(records, results, localization_targets=targets)
    report['localization_ambiguity'] = {
        'policy': 'Original per-ID localization fields are unchanged. This additional view accepts any supplied timestamp within the exact same SHA-256 group; it is an annotation ambiguity, not a confirmed event interval.',
        'conflicting_exact_content_groups': conflicts,
        'median_acceptable_gt_error_sec': ambiguity_report['median_localization_error_sec'],
        'matched_positive_videos': ambiguity_report['localization_matched_positive_videos'],
    }
    report['unique_content'] = {
        **_summarize_rows(representatives, unique_results, localization_targets=targets),
        'unit': 'exact SHA-256 content; no SHA means one separate ID',
        'selection_policy': 'Lowest numeric ID is canonical. Its prediction is used even if an alias succeeded; no voting or best-result selection. Conflicting group labels are excluded as failures.',
        'localization_policy': 'Accept the set of original supplied timestamps for the exact same content.',
        'groups': group_details,
    }
    return report


def run_evaluation(manifest: dict, splits: dict, config: dict, output_dir: str | Path,
                   *, partition: str = 'validation', limit: int | None = None,
                   run_metadata: dict | None = None) -> dict:
    wanted = set(splits['ids'][partition])
    records = [r for r in manifest['records'] if r['id'] in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        records = records[:limit]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint, fingerprint_components = build_run_fingerprint(config, run_metadata=run_metadata)
    metadata = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'partition': partition, 'config_fingerprint': fingerprint,
        'fingerprint_components': fingerprint_components,
        'manifest_sha256': _json_digest(manifest), 'splits_sha256': _json_digest(splits),
        'blind_holdout': False,
        'holdout_disclosure': 'All descriptions were reviewed. X-26 was used to tune the earlier prototype and X-46 was previously run there; current validation/test are not independent blind holdouts.',
        'known_prior_inference_ids': list(KNOWN_PRIOR_INFERENCE_IDS),
        'known_prior_inference_ids_in_partition': [record['id'] for record in records if record['id'] in KNOWN_PRIOR_INFERENCE_IDS],
    }

    def report_with_metadata():
        return {**summarize(records, results), 'run_metadata': metadata}

    results = {}
    for index, record in enumerate(records, 1):
        print(f'[{index}/{len(records)}] {record["id"]}', flush=True)
        try:
            result = analyze_video(record['video_path'], config)
            checked, failure = _checked_prediction(record, result)
            if failure:
                result = {'status': 'failed', 'error': {'type': 'MalformedPrediction' if failure == 'malformed_prediction' else failure,
                                                       'message': 'Analysis did not return a valid completed result.'}}
            else:
                result = checked
        except Exception as exc:
            result = {'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc)[:500]}}
        results[record['id']] = result
        saved = {'video_sha256': record['sha256'], 'config_fingerprint': fingerprint,
                 'run_metadata': metadata, 'result': result}
        (output / f'{record["id"]}.json').write_text(json.dumps(saved, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        (output / 'metrics.json').write_text(json.dumps(report_with_metadata(), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    summary = report_with_metadata()
    (output / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return summary
