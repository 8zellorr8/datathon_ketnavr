"""Audit saved evaluation evidence locally; never run the model or print images.

The manifest ID lookup is used only for this offline audit. No annotations,
filenames, or expected outcomes are passed to an inference function.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import uuid


ROOT = Path(__file__).resolve().parent
TIMING_METHOD = 'artifact timestamps, estimate'


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON property')
        result[key] = value
    return result


def _read_json(path: Path):
    def reject_constant(_value):
        raise ValueError('Non-finite JSON constant')
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def _utc_timestamp(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_helper():
    spec = importlib.util.spec_from_file_location('_forte_evidence_measurement',
                                                  ROOT / 'Measure-Analysis.py')
    if spec is None or spec.loader is None:
        raise ImportError('Evidence utility cannot be imported')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_evidence


def _error(code: str, error: BaseException | None = None) -> dict:
    # Exception text can contain paths, model descriptions, or encoded frames.
    result = {'code': code}
    if error is not None:
        result['class'] = type(error).__name__
    return result


def _timings(artifacts: list[dict]) -> None:
    """Attach a sequential wall-time estimate, grouped by evaluation run."""
    previous_by_run = {}
    for item in sorted(artifacts, key=lambda row: (row['_mtime_ns'], row['id'])):
        wrapper = item.get('_wrapper', {})
        metadata = wrapper.get('run_metadata', {}) if isinstance(wrapper, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        started = metadata.get('started_at_utc')
        run_key = (str(started), str(metadata.get('partition')),
                   str(wrapper.get('config_fingerprint')) if isinstance(wrapper, dict) else '')
        previous = previous_by_run.get(run_key)
        baseline = previous['_mtime_ns'] / 1e9 if previous is not None else _utc_timestamp(started)
        end = item['_mtime_ns'] / 1e9
        estimate = end - baseline if baseline is not None else None
        valid = estimate is not None and math.isfinite(estimate) and estimate >= 0
        item['wall_time'] = {
            'method': TIMING_METHOD,
            'estimate_sec': round(estimate, 3) if valid else None,
            'baseline': 'previous_prediction_mtime_utc' if previous is not None else 'started_at_utc',
            'previous_id': previous['id'] if previous is not None else None,
            'artifact_mtime_utc': _iso(end),
        }
        if not valid:
            item['wall_time']['unavailable_reason'] = 'missing_or_inconsistent_artifact_timestamps'
        previous_by_run[run_key] = item


def audit(directory: Path, manifest_path: Path, config_path: Path, *,
          check_evidence=None, validate_result=None) -> dict:
    """Return a numeric audit. Injected helpers support CPU-only mock checks."""
    manifest = _read_json(manifest_path)
    config = _read_json(config_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get('records'), list):
        raise ValueError('Invalid manifest')
    if not isinstance(config, dict):
        raise ValueError('Invalid config')
    max_side = config.get('frame_max_side', 560)
    if type(max_side) is not int or not 128 <= max_side <= 1280:
        raise ValueError('Invalid frame size')
    records = {}
    for record in manifest['records']:
        if not isinstance(record, dict) or not isinstance(record.get('id'), str):
            raise ValueError('Invalid manifest record')
        if record['id'] in records:
            raise ValueError('Duplicate manifest ID')
        records[record['id']] = record
    if check_evidence is None:
        check_evidence = _measurement_helper()
    if validate_result is None:
        from forte_ai.server import _validate_analysis_result
        validate_result = _validate_analysis_result

    artifacts = []
    for path in directory.glob('X-*.json'):
        if not path.is_file():
            continue
        item = {'id': path.stem, '_mtime_ns': path.stat().st_mtime_ns,
                'status': 'invalid', 'evidence_checks': [], 'errors': []}
        artifacts.append(item)
        try:
            wrapper = _read_json(path)
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get('result'), dict):
                raise ValueError('Expected nested result')
            item['_wrapper'] = wrapper
            result = wrapper['result']
            status = result.get('status')
            if status not in ('done', 'failed'):
                raise ValueError('Expected a terminal result')
            item['status'] = status
            if status == 'failed':
                # An analyzer failure is counted separately; it is no clean verdict.
                continue
            validate_result(result)
            json.dumps(result, allow_nan=False)
            item['verdict'] = result['verdict']
            item['event_count'] = len(result['violations'])
            reported_elapsed = result.get('analysis', {}).get('elapsed_sec')
            if (type(reported_elapsed) in (int, float) and math.isfinite(reported_elapsed)
                    and reported_elapsed >= 0):
                item['reported_analysis_elapsed_sec'] = reported_elapsed
            record = records.get(path.stem)
            if record is None:
                raise ValueError('Prediction ID is absent from manifest')
            video_value = record.get('video_path')
            expected_hash = record.get('sha256')
            if (not isinstance(video_value, str) or not isinstance(expected_hash, str)
                    or re.fullmatch(r'[0-9a-fA-F]{64}', expected_hash) is None):
                raise ValueError('Invalid video reference')
            video = Path(video_value).expanduser()
            if not video.is_absolute():
                video = ROOT / video
            video = video.resolve()
            if not video.is_file():
                raise FileNotFoundError('Original video is unavailable')
            item['original_video_sha256_matches_manifest'] = _sha256(video) == expected_hash.lower()
            item['artifact_video_sha256_matches_manifest'] = wrapper.get('video_sha256') == expected_hash.lower()
            duration = result.get('analysis', {}).get('duration_sec')
            if duration is not None and (type(duration) not in (int, float)
                                         or not math.isfinite(duration) or duration <= 0):
                raise ValueError('Invalid duration')
            for event_index, event in enumerate(result['violations']):
                for frame_index, frame in enumerate(event['evidence_frames']):
                    entry = {'event_index': event_index, 'frame_index': frame_index,
                             'timestamp_sec': frame['timestamp_sec']}
                    item['evidence_checks'].append(entry)
                    if duration is not None and (event['timecode_sec'] > duration
                                                 or frame['timestamp_sec'] > duration):
                        entry.update(pts_matches=False, jpeg_matches=False,
                                     error=_error('evidence_outside_video_duration'))
                        continue
                    # Audit one frame at a time so one decode error does not skip
                    # the other saved frames. Only FFmpeg receives the video path.
                    single = {**result, 'violations': [{**event, 'evidence_frames': [frame]}]}
                    try:
                        checks = check_evidence(single, video, {'frame_max_side': max_side})
                        if not isinstance(checks, list) or len(checks) != 1:
                            raise ValueError('Unexpected evidence-check result')
                        checked = checks[0]
                        if (type(checked.get('pts_matches')) is not bool
                                or type(checked.get('jpeg_matches')) is not bool
                                or type(checked.get('redecoded_timestamp_sec')) not in (int, float)
                                or not math.isfinite(checked['redecoded_timestamp_sec'])):
                            raise ValueError('Invalid evidence-check result')
                        entry.update(redecoded_timestamp_sec=checked['redecoded_timestamp_sec'],
                                     pts_matches=checked['pts_matches'],
                                     jpeg_matches=checked['jpeg_matches'])
                    except Exception as error:
                        entry['error'] = _error('evidence_utility_failed', error)
            item['schema_valid'] = True
        except Exception as error:
            item['errors'].append(_error('artifact_audit_failed', error))

    _timings(artifacts)
    artifacts.sort(key=lambda row: (int(row['id'].split('-')[1])
                                   if re.fullmatch(r'X-\d+', row['id']) else sys.maxsize, row['id']))
    mismatches, utility_failures = [], []
    for item in artifacts:
        mismatch = (item.get('original_video_sha256_matches_manifest') is False
                    or item.get('artifact_video_sha256_matches_manifest') is False
                    or any(check.get('pts_matches') is False or check.get('jpeg_matches') is False
                           for check in item['evidence_checks']))
        utility_failure = bool(item['errors']) or any('error' in check for check in item['evidence_checks'])
        item['evidence_valid'] = (not mismatch and not utility_failure
                                  if item['status'] == 'done' else None)
        if mismatch:
            mismatches.append(item['id'])
        if utility_failure:
            utility_failures.append(item['id'])
        item.pop('_wrapper', None)
        item.pop('_mtime_ns', None)
    checks = [check for item in artifacts for check in item['evidence_checks']]
    counts = {
        'prediction_files': len(artifacts),
        'done': sum(item['status'] == 'done' for item in artifacts),
        'failed': sum(item['status'] == 'failed' for item in artifacts),
        'invalid': sum(item['status'] == 'invalid' for item in artifacts),
        'done_without_evidence': sum(item['status'] == 'done' and item.get('event_count') == 0
                                     for item in artifacts),
        'evidence_frames': len(checks),
        'pts_matches': sum(check.get('pts_matches') is True for check in checks),
        'jpeg_matches': sum(check.get('jpeg_matches') is True for check in checks),
        'mismatch_videos': len(mismatches),
        'utility_failure_videos': len(utility_failures),
    }
    return {
        'schema_version': 1,
        'audit_status': 'failed' if mismatches or utility_failures else 'passed',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved evidence integrity only; this does not verify the inferred event or legal conclusion.',
        'decode': {'method': 'FFmpeg re-extraction through Measure-Analysis.check_evidence',
                   'frame_max_side': max_side, 'pts_tolerance_sec': 0.001,
                   'jpeg_comparison': 'SHA-256 equality of original and re-extracted JPEG bytes'},
        'timing': {
            'method': TIMING_METHOD,
            'assumption': 'Prediction files were written sequentially and have not been copied or modified.',
            'limitations': 'Estimates include startup, writes, between-video work and pauses; they are not measured model latency.',
        },
        'counts': counts,
        'failed_ids': [item['id'] for item in artifacts if item['status'] == 'failed'],
        'mismatch_ids': mismatches,
        'utility_failure_ids': utility_failures,
        'videos': artifacts,
    }


def _write_report(directory: Path, report: dict) -> None:
    temporary = directory / ('.evidence-audit-' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding='utf-8')
        temporary.replace(directory / 'evidence_audit.json')
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/validation')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/local.json')
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    # This utility writes only its report inside this project workspace.
    if not directory.is_relative_to(ROOT):
        print(json.dumps({'audit_status': 'failed', 'utility_failure_count': 1,
                          'error': {'code': 'output_outside_project_workspace'}}))
        return 1
    try:
        directory.mkdir(parents=True, exist_ok=True)
        report = audit(directory, args.manifest.expanduser().resolve(),
                       args.config.expanduser().resolve())
    except Exception as error:
        report = {'schema_version': 1, 'audit_status': 'failed', 'counts': {},
                  'utility_failure_count': 1, 'error': _error('audit_utility_failed', error)}
    try:
        _write_report(directory, report)
    except Exception as error:
        print(json.dumps({'audit_status': 'failed', 'utility_failure_count': 1,
                          'error': _error('report_write_failed', error)}))
        return 1
    print(json.dumps({'audit_status': report['audit_status'], **report.get('counts', {}),
                      **({'utility_failure_count': report['utility_failure_count']}
                         if 'utility_failure_count' in report else {})}), flush=True)
    return 0 if report['audit_status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
