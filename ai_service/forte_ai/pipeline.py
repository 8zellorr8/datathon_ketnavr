"""Temporal video review through a locally installed vision-language model."""
from __future__ import annotations

import json
import math
import re
from enum import Enum
from pathlib import Path
import time

from .local_vlm import analyze_segment, LocalVLMOutputError, assess_room_door
from .video import extract_frame, probe_video, sample_times


class AnalysisErrorCode(str, Enum):
    ANALYSIS_FAILED = 'analysis_failed'
    TIME_BUDGET_EXCEEDED = 'time_budget_exceeded'
    CANDIDATE_BUDGET_EXCEEDED = 'candidate_budget_exceeded'
    EVIDENCE_BUDGET_EXCEEDED = 'evidence_budget_exceeded'
    INVALID_ANALYSIS_CONFIGURATION = 'invalid_analysis_configuration'
    INVALID_RULE_CATALOGUE = 'invalid_rule_catalogue'
    NO_VERIFIED_RULES = 'no_verified_rules'
    VIDEO_DURATION_EXCEEDED = 'video_duration_exceeded'


class AnalysisError(RuntimeError):
    """Internal failure with a fixed diagnostic code safe for HTTP/log output."""

    def __init__(self, message: str, *, safe_code=AnalysisErrorCode.ANALYSIS_FAILED):
        super().__init__(message)
        if type(safe_code) is AnalysisErrorCode:
            self.safe_code = safe_code
        elif type(safe_code) is str:
            try:
                self.safe_code = AnalysisErrorCode(safe_code)
            except ValueError:
                self.safe_code = AnalysisErrorCode.ANALYSIS_FAILED
        else:
            self.safe_code = AnalysisErrorCode.ANALYSIS_FAILED


DEFAULTS = {
    'base_url': 'http://127.0.0.1:11434',
    'model': 'qwen3-vl:2b-instruct-q4_K_M',
    'coarse_step_sec': 4.0,
    'frames_per_window': 6,
    'overlap_frames': 1,
    # A larger window is opt-in until measured on the target hardware.
    'long_video_threshold_sec': 60.0,
    'long_video_frames_per_window': None,
    'long_video_overlap_frames': None,
    'num_ctx': 8192,
    'image_layout': 'contact_sheet',
    'frames_per_sheet': 6,
    'contact_sheet_max_side': 1250,
    'frame_max_side': 560,
    'max_duration_sec': 180.0,
    'budget_sec': 270.0,
    'request_timeout_sec': 90.0,
    'candidate_confidence': 0.6,
    'confirmed_confidence': 0.8,
    # None reviews every candidate within the shared deadline. An explicit
    # nonnegative integer remains available as a fail-closed deployment cap.
    'max_refinements': None,
    'refine_radius_sec': 2.0,
    'refine_step_sec': 0.5,
    'runtime_dir': 'data/runtime',
    # Visually observable bank-specific review signals have no verified legal
    # qualification. Keep their catalogue separate from verified public rules.
    'visual_checks_path': None,
    'scene_state_scan': False,
}


def load_config(path: str | Path) -> dict:
    location = Path(path).resolve()
    config = {**DEFAULTS, **json.loads(location.read_text(encoding='utf-8'))}
    rules_path = Path(config.get('rules_path', 'rules.json'))
    config['rules_path'] = str((location.parent / rules_path).resolve())
    if config.get('frontend_dir') is not None:
        config['frontend_dir'] = str((location.parent / Path(config['frontend_dir'])).resolve())
    if config.get('visual_checks_path') is not None:
        config['visual_checks_path'] = str((location.parent / Path(config['visual_checks_path'])).resolve())
    runtime_path = Path(config['runtime_dir'])
    config['runtime_dir'] = str((location.parent.parent / runtime_path).resolve())
    return config


def _time_left(deadline: float, cap: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0.1:
        raise AnalysisError('Analysis exceeded its processing budget', safe_code=AnalysisErrorCode.TIME_BUDGET_EXCEEDED)
    return min(cap, remaining)


def _windows(frames: list[dict], size: int, overlap: int):
    if not 2 <= size <= 12 or not 0 <= overlap < size:
        raise AnalysisError('Invalid window size or overlap', safe_code=AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION)
    for start in range(0, len(frames), size - overlap):
        chunk = frames[start:start + size]
        if start and len(chunk) <= overlap:
            break
        yield chunk


def _window_settings(config: dict, duration_sec: float) -> tuple[int, int]:
    size, overlap = config['frames_per_window'], config['overlap_frames']
    threshold = config['long_video_threshold_sec']
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or threshold <= 0):
        raise AnalysisError('Invalid long-video threshold',
                            safe_code=AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION)
    if duration_sec > threshold and config['long_video_frames_per_window'] is not None:
        size = config['long_video_frames_per_window']
        if config['long_video_overlap_frames'] is not None:
            overlap = config['long_video_overlap_frames']
    if (type(size) is not int or type(overlap) is not int
            or not 2 <= size <= 12 or not 0 <= overlap < size):
        raise AnalysisError('Invalid window size or overlap',
                            safe_code=AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION)
    return size, overlap


def _read_rules(path: str | Path) -> list[dict]:
    rules = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rules, list):
        raise AnalysisError('Rule catalogue must be a JSON list', safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
    enabled = [r for r in rules if r.get('enabled') and r.get('source', {}).get('verified')]
    if not enabled:
        raise AnalysisError('No enabled rules with verified sources are configured', safe_code=AnalysisErrorCode.NO_VERIFIED_RULES)
    identifiers = [r.get('id') for r in enabled]
    if len(set(identifiers)) != len(identifiers) or not all(identifiers):
        raise AnalysisError('Rule IDs must be nonempty and unique', safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
    return [{**r, 'rule_id': r['id']} for r in enabled]


def _read_visual_checks(path: str | Path | None) -> list[dict]:
    """Opt-in visual review signals, never a source of legal authority."""
    if path is None:
        return []
    checks = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(checks, list) or len(checks) > 16:
        raise AnalysisError('Visual checks must be a bounded JSON list',
                            safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
    result, identifiers = [], set()
    for check in checks:
        if not isinstance(check, dict) or type(check.get('enabled')) is not bool:
            raise AnalysisError('Invalid visual check', safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
        if not check['enabled']:
            continue
        identifier = check.get('id')
        if (not isinstance(identifier, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', identifier)
                or identifier in identifiers or not isinstance(check.get('name'), str)
                or not check['name'].strip()):
            raise AnalysisError('Invalid or duplicate visual check identifier',
                                safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
        for key in ('visual_requirements', 'counterexamples'):
            values = check.get(key)
            if (not isinstance(values, list) or not 1 <= len(values) <= 12
                    or any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in values)):
                raise AnalysisError('Invalid visual check criteria',
                                    safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
        minimum = check.get('min_evidence_frames', 2)
        span = check.get('min_evidence_span_sec', 1.0)
        if (type(minimum) is not int or not 2 <= minimum <= 3
                or type(span) not in (int, float) or not math.isfinite(span) or not 0 < span <= 30):
            raise AnalysisError('Invalid visual check evidence requirements',
                                safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
        identifiers.add(identifier)
        # Explicitly whitelist fields: neither a supplied source nor a forged
        # verified flag can promote a visual observation into a verified rule.
        result.append({
            'id': identifier, 'rule_id': identifier, 'name': check['name'].strip(),
            'visual_requirements': check['visual_requirements'], 'counterexamples': check['counterexamples'],
            'min_evidence_frames': minimum, 'min_evidence_span_sec': float(span),
            'source': None, 'verification_status': 'needs_regulation_review',
            'detection_basis': 'visual_observation',
        })
    return result


def _nearby_times(center: float, duration: float, radius: float, step: float) -> list[float]:
    start, end = max(0, center - radius), min(duration - 0.15, center + radius)
    if step <= 0 or radius <= 0 or end < start:
        raise AnalysisError('Invalid refinement interval', safe_code=AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION)
    times = sample_times(end - start + 0.15, step)
    return sorted(set(round(start + t, 4) for t in times))


def _checked_segment(frames: list[dict], *, config: dict, deadline: float,
                     rules: list[dict], context: str) -> dict:
    """Retry one malformed model answer, never a resource/backend failure.

    No answer or label is supplied to the retry. Every attempt sees the same
    source frames, rules and absolute timestamps and must pass the full schema.
    """
    attempt_times = []
    for attempt in range(2):
        current_context = context
        if attempt:
            current_context += (
                '\nПовторная техническая проверка: прошлый ответ не прошёл проверку формата. '
                'Используй только заданные индексы кадров. В каждом событии '
                'evidence_frame_indices должны быть уникальны, содержать frame_index '
                'и включать не более трёх кадров. Не добавляй событие ради заполнения ответа.'
            )
        attempt_started = time.monotonic()
        try:
            response = analyze_segment(
                frames, rules=rules, context=current_context,
                base_url=config['base_url'], model=config['model'],
                timeout_sec=_time_left(deadline, float(config['request_timeout_sec'])),
                num_ctx=int(config['num_ctx']), image_layout=config['image_layout'],
                frames_per_sheet=int(config['frames_per_sheet']),
                contact_sheet_max_side=int(config['contact_sheet_max_side']),
            )
        except LocalVLMOutputError:
            if attempt:
                raise
            attempt_times.append(round(time.monotonic() - attempt_started, 3))
            continue
        attempt_times.append(round(time.monotonic() - attempt_started, 3))
        response['inference_meta'] = {
            **response.get('inference_meta', {}),
            'attempt_count': attempt + 1,
            'attempts_elapsed_sec': attempt_times,
        }
        return response
    raise LocalVLMOutputError('Local model output remained invalid after a bounded retry.')


_CASH_RULE_IDS = frozenset({
    'unattended_cash', 'cash_not_secured_between_clients', 'count_after_client_exit',
    'cash_issued_without_count', 'uncounted_unsealed_cash',
})
_DOOR_CHECK_ID = 'door_left_open_while_working'


def _scene_candidates(states: list[dict], frames: list[dict], rule: dict,
                      confidence: float) -> list[dict]:
    """An open doorway during passage is not persistent open-door work."""
    run = []
    for state in states:
        if (state['door_state'] == 'open' and state['work_state'] == 'working'
                and state['confidence'] >= confidence):
            run.append(state)
            first, last = run[0], run[-1]
            span = frames[last['frame_index']]['timestamp_sec'] - frames[first['frame_index']]['timestamp_sec']
            if len(run) >= rule['min_evidence_frames'] and span >= rule['min_evidence_span_sec']:
                witnesses = [first['frame_index'], last['frame_index']]
                if rule['min_evidence_frames'] == 3:
                    witnesses.insert(1, run[len(run) // 2]['frame_index'])
                return [{
                    'rule_id': rule['id'], 'frame_index': first['frame_index'],
                    'timecode_sec': frames[first['frame_index']]['timestamp_sec'],
                    'description': 'Сотрудник работает у стола, а входная дверь кабинета остаётся открытой на нескольких кадрах.',
                    'confidence': min(item['confidence'] for item in run),
                    'evidence_frame_indices': witnesses,
                }]
        else:
            # Closed/uncertain/transit frames break the observed interval.
            run = []
    return []


def _checked_scene(frames: list[dict], *, config: dict, deadline: float) -> dict:
    attempt_times = []
    for attempt in range(2):
        attempt_started = time.monotonic()
        try:
            response = assess_room_door(
                frames, base_url=config['base_url'], model=config['model'],
                timeout_sec=_time_left(deadline, float(config['request_timeout_sec'])),
                num_ctx=int(config['num_ctx']), image_layout=config['image_layout'],
                frames_per_sheet=int(config['frames_per_sheet']),
                contact_sheet_max_side=int(config['contact_sheet_max_side']),
            )
        except LocalVLMOutputError:
            if attempt:
                raise
            attempt_times.append(round(time.monotonic() - attempt_started, 3))
            continue
        attempt_times.append(round(time.monotonic() - attempt_started, 3))
        response['inference_meta'] = {**response.get('inference_meta', {}),
                                     'attempt_count': attempt + 1, 'attempts_elapsed_sec': attempt_times}
        return response
    raise LocalVLMOutputError('Scene state output remained invalid after a bounded retry.')


def analyze_video(video_path: str | Path, config: dict, *, deadline: float | None = None) -> dict:
    """No workbook, labels, filenames or annotation text is used in inference."""
    started = time.monotonic()
    cfg = {**DEFAULTS, **config}
    deadline = min(deadline or float('inf'), started + float(cfg['budget_sec']))
    _time_left(deadline, 1)
    normative_rules = _read_rules(cfg['rules_path'])
    checks = _read_visual_checks(cfg['visual_checks_path'])
    rules = normative_rules + checks
    if len({r['id'] for r in rules}) != len(rules):
        raise AnalysisError('Rule and visual check identifiers must be distinct',
                            safe_code=AnalysisErrorCode.INVALID_RULE_CATALOGUE)
    rule_by_id = {r['id']: r for r in rules}
    visual_rules = [{'rule_id': r['id'], 'visual_requirements': r.get('visual_requirements', []),
                     'counterexamples': r.get('counterexamples', [])} for r in rules]
    visual_by_id = {r['rule_id']: r for r in visual_rules}
    if type(cfg['scene_state_scan']) is not bool:
        raise AnalysisError('scene_state_scan must be boolean',
                            safe_code=AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION)
    scene_scan = cfg['scene_state_scan'] and _DOOR_CHECK_ID in rule_by_id
    metadata = probe_video(video_path, deadline=deadline)
    if metadata['duration_sec'] > float(cfg['max_duration_sec']):
        raise AnalysisError('Video exceeds configured maximum duration', safe_code=AnalysisErrorCode.VIDEO_DURATION_EXCEEDED)
    window_size, overlap = _window_settings(cfg, metadata['duration_sec'])
    max_refinements = cfg['max_refinements']
    if max_refinements is not None and (type(max_refinements) is not int or max_refinements < 0):
        raise AnalysisError('max_refinements must be null or a nonnegative integer',
                            safe_code=AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION)
    coarse_times = sample_times(metadata['duration_sec'], float(cfg['coarse_step_sec']))
    # Keep only the current window's JPEGs. Candidates retain PTS values, not
    # image payloads, so memory does not grow with the duration of the scan.
    frame_buffer = {}
    coarse_frames_decoded = 0
    refinement_frames_decoded = 0
    peak_buffer_frames = 0
    context = ''
    candidates = []
    observation_count = 0
    calls = 0
    scan_window_count = 0
    request_stats = []
    cash_prerequisite_exclusions = 0
    for window_times in _windows(coarse_times, window_size, overlap):
        frame_buffer = {t: frame_buffer[t] for t in window_times if t in frame_buffer}
        for t in window_times:
            if t not in frame_buffer:
                frame_buffer[t] = extract_frame(video_path, t, max_side=int(cfg['frame_max_side']), deadline=deadline)
                coarse_frames_decoded += 1
        chunk = [frame_buffer[t] for t in window_times]
        peak_buffer_frames = max(peak_buffer_frames, len(chunk))
        scan_rules = visual_rules
        segment_candidates = []
        if scene_scan:
            scene = _checked_scene(chunk, config=cfg, deadline=deadline)
            calls += scene['inference_meta']['attempt_count']
            request_stats.append({'phase': 'scan_scene', **scene['inference_meta']})
            segment_candidates.extend(_scene_candidates(scene['frames'], chunk, rule_by_id[_DOOR_CHECK_ID],
                                                        float(cfg['confirmed_confidence'])))
            scan_rules = [r for r in visual_rules if r['rule_id'] != _DOOR_CHECK_ID]
            # This gate concerns only positive visual support in these frames.
            # Unknown/occluded cash or any visible cash keeps all money rules.
            if scene['frames'] and all(s['cash_visible'] == 'no' and s['confidence'] >= .9
                                       for s in scene['frames']):
                cash_prerequisite_exclusions += sum(r['rule_id'] in _CASH_RULE_IDS for r in scan_rules)
                scan_rules = [r for r in scan_rules if r['rule_id'] not in _CASH_RULE_IDS]
        if scan_rules:
            response = _checked_segment(chunk, config=cfg, deadline=deadline,
                                        rules=scan_rules, context=context)
            calls += response['inference_meta']['attempt_count']
            request_stats.append({'phase': 'scan', **response.get('inference_meta', {})})
            observation_count += len(response['observations'])
            context = response['summary'][:1800]
            segment_candidates.extend(response['candidate_violations'])
        scan_window_count += 1
        for candidate in segment_candidates:
            if candidate['confidence'] >= float(cfg['candidate_confidence']):
                candidates.append({**candidate, '_source_times':
                                   [float(chunk[i]['timestamp_sec']) for i in candidate['evidence_frame_indices']]})
        # Do not retain the previous window through the next decode cycle.
        del chunk
    frame_buffer.clear()

    # Overlapping windows may describe the same event. Merge by rule and time.
    merged = []
    for candidate in sorted(candidates, key=lambda c: -c['confidence']):
        match = next((old for old in merged if candidate['rule_id'] == old['rule_id'] and
                      abs(candidate['timecode_sec'] - old['timecode_sec']) <= 6), None)
        if match is None:
            merged.append(candidate)
        else:
            match['_source_times'] = sorted(set(match['_source_times'] + candidate['_source_times']))
    if max_refinements is not None and len(merged) > max_refinements:
        raise AnalysisError('Too many candidate events for the configured review budget; increase max_refinements',
                            safe_code=AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED)

    violations = []
    refinement_count = 0
    for candidate in sorted(merged, key=lambda c: c['timecode_sec']):
        # Release the previous refinement window before decoding the next one.
        nearby = []
        by_time = {}
        source_times = sorted(set(candidate['_source_times']))
        if len(source_times) > 24:
            raise AnalysisError('Refinement evidence exceeds the local model frame budget',
                                safe_code=AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED)
        # Re-decode the original causal witnesses. FFmpeg reports rounded PTS;
        # retry infinitesimally before a witness only if seeking that rounded
        # value selected the following frame. Never substitute a different PTS.
        for t in source_times:
            frame = extract_frame(video_path, t, max_side=int(cfg['frame_max_side']), deadline=deadline)
            refinement_frames_decoded += 1
            if abs(float(frame['timestamp_sec']) - t) > 0.000001:
                frame = extract_frame(video_path, max(0.0, t - 0.00001),
                                      max_side=int(cfg['frame_max_side']), deadline=deadline)
                refinement_frames_decoded += 1
            if abs(float(frame['timestamp_sec']) - t) > 0.000001:
                raise AnalysisError('Could not reproduce an original causal evidence frame')
            by_time[frame['timestamp_sec']] = frame
        for t in _nearby_times(float(candidate['timecode_sec']), metadata['duration_sec'],
                               float(cfg['refine_radius_sec']), float(cfg['refine_step_sec'])):
            if any(abs(float(actual) - t) <= 0.000001 for actual in by_time):
                continue
            frame = extract_frame(video_path, t, max_side=int(cfg['frame_max_side']), deadline=deadline)
            refinement_frames_decoded += 1
            by_time[frame['timestamp_sec']] = frame
            if len(by_time) > 24:
                raise AnalysisError('Refinement evidence exceeds the local model frame budget',
                                    safe_code=AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED)
        nearby = [by_time[t] for t in sorted(by_time)]
        peak_buffer_frames = max(peak_buffer_frames, len(nearby))
        rule = rule_by_id[candidate['rule_id']]
        if scene_scan and rule['id'] == _DOOR_CHECK_ID:
            response = _checked_scene(nearby, config=cfg, deadline=deadline)
            response['candidate_violations'] = _scene_candidates(response['frames'], nearby, rule,
                                                                float(cfg['confirmed_confidence']))
            phase = 'refine_scene'
        else:
            response = _checked_segment(
                nearby, config=cfg, deadline=deadline, rules=[visual_by_id[rule['id']]], context=(
                'Проверь гипотезу по более частым кадрам. Исходная гипотеза может быть ошибочной. '
                'Не подтверждай её автоматически. Укажи видимое действие; если подтверждения нет, '
                'верни пустой candidate_violations. Гипотеза: ' + candidate['description'][:600]
                ),
            )
            phase = 'refine'
        calls += response['inference_meta']['attempt_count']
        refinement_count += 1
        request_stats.append({'phase': phase, 'rule_id': rule['id'], **response.get('inference_meta', {})})
        confirmations = [c for c in response['candidate_violations']
                         if c['rule_id'] == rule['id'] and c['confidence'] >= float(cfg['confirmed_confidence'])]
        if not confirmations:
            continue
        best = max(confirmations, key=lambda c: c['confidence'])
        indices = [best['frame_index']] + [i for i in best['evidence_frame_indices'] if i != best['frame_index']]
        if not 1 <= len(indices) <= 3:
            # The adapter schema already enforces this. Preserve the same
            # fail-closed boundary if another analyzer is injected later.
            raise AnalysisError('Confirmed evidence must contain 1 to 3 frames; causal witnesses cannot be silently truncated',
                                safe_code=AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED)
        evidence = [nearby[i] for i in indices]
        is_visual_check = rule.get('verification_status') == 'needs_regulation_review'
        if is_visual_check:
            pts = sorted({float(frame['timestamp_sec']) for frame in evidence})
            if (len(pts) < rule['min_evidence_frames']
                    or pts[-1] - pts[0] < rule['min_evidence_span_sec']):
                continue
        violations.append({
            'type': rule['name'], 'timecode_sec': float(best['timecode_sec']),
            'description': best['description'], 'evidence_frames': evidence,
            'rule_id': rule['id'], 'rule_source': rule['source'],
            'verification_status': 'needs_regulation_review' if is_visual_check else 'normative_source_verified',
            'detection_basis': 'visual_observation' if is_visual_check else 'verified_rule',
            'requires_human_review': True,
            **({'regulatory_note': 'Зафиксировано визуальное событие. Применимый запрет и внутренний регламент банка не подтверждены; требуется проверка специалистом.'}
               if is_visual_check else {}),
        })
    _time_left(deadline, 1)
    return {
        'status': 'done', 'verdict': 'violation' if violations else 'no_violation',
        'violations': violations,
        'analysis': {
            'backend': 'local_ollama', 'model': cfg['model'],
            'duration_sec': metadata['duration_sec'], 'elapsed_sec': round(time.monotonic() - started, 3),
            'sampled_frames': len(coarse_times), 'model_calls': calls,
            'effective_frames_per_window': window_size, 'effective_overlap_frames': overlap,
            'scan_window_count': scan_window_count,
            'candidate_count': len(candidates), 'merged_candidate_count': len(merged),
            'refinement_count': refinement_count,
            'scan_complete': True, 'refinements_complete': refinement_count == len(merged),
            # This counts the decoded working window, excluding final evidence
            # already attached to results and the model adapter's own copies.
            'peak_buffer_frames': peak_buffer_frames,
            'coarse_frames_decoded': coarse_frames_decoded,
            'refinement_frames_decoded': refinement_frames_decoded,
            'num_ctx': int(cfg['num_ctx']), 'image_layout': cfg['image_layout'],
            'frames_per_sheet': int(cfg['frames_per_sheet']),
            'contact_sheet_max_side': int(cfg['contact_sheet_max_side']),
            'model_requests': request_stats,
            'observation_count': observation_count,
            'enabled_rule_ids': list(rule_by_id),
            'verified_rule_ids': [r['id'] for r in normative_rules],
            'visual_check_ids': [r['id'] for r in checks],
            'coverage': 'configured_checks_only',
            'scene_state_scan': scene_scan,
            'cash_prerequisite_exclusions': cash_prerequisite_exclusions,
            'version': 'prototype_0.1',
        },
    }
