"""Measure one real local upload without printing frames, descriptions or API keys."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('local_measurement', ROOT / 'Measure-Analysis.py')
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


def run_api_measurement(args, output: Path) -> int:
    from forte_ai.local_vlm import _local_origin, _NoRedirect
    from forte_ai.pipeline import load_config
    origin = _local_origin(args.api_url)
    key = os.environ.get('FORTE_API_KEY')
    if not key:
        key = (ROOT / 'data/runtime/api-key.txt').read_text(encoding='ascii').strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def request(path, body=None, content_type=None, authorized=True):
        headers = {'X-API-Key': key} if authorized else {}
        if content_type:
            headers['Content-Type'] = content_type
        req = urllib.request.Request(origin + path, data=body, headers=headers)
        with opener.open(req, timeout=15) as response:
            return response.status, json.load(response)

    unauthorized_status = None
    try:
        request('/api/v1/jobs/00000000000000000000000000000000', authorized=False)
    except urllib.error.HTTPError as error:
        unauthorized_status = error.code
        error.close()
    if unauthorized_status != 401:
        raise RuntimeError('API must reject unauthenticated job requests with HTTP 401')
    boundary = 'forte-' + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="upload.mov"\r\n'
            'Content-Type: video/quicktime\r\n\r\n').encode('ascii')
    body += args.video.read_bytes() + f'\r\n--{boundary}--\r\n'.encode('ascii')
    started = time.monotonic()
    status, job = request('/api/v1/analyze', body, f'multipart/form-data; boundary={boundary}')
    if status != 202 or job.get('status') != 'processing':
        raise RuntimeError('Upload did not start an asynchronous job')
    telemetry, statuses = [], ['processing']
    next_log = 0
    while True:
        telemetry.append(measurement.snapshot(started))
        status, result = request('/api/v1/jobs/' + job['job_id'])
        if result.get('status') != statuses[-1]:
            statuses.append(result.get('status'))
        elapsed = time.monotonic() - started
        if elapsed >= next_log:
            print(json.dumps({'status': result.get('status'), 'elapsed_sec': round(elapsed, 1)}), flush=True)
            next_log = elapsed + 20
        if result.get('status') in ('done', 'failed'):
            break
        if elapsed > 310:
            raise RuntimeError('Job did not become terminal within 310 seconds')
        time.sleep(1)
    (output / 'prediction.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'status': result['status'], 'verdict': result.get('verdict'),
              'elapsed_sec': round(time.monotonic() - started, 3), 'http_upload_status': 202,
              'http_poll_status': status, 'http_without_key': unauthorized_status,
              'status_sequence': statuses, 'uploaded_name': 'upload.mov',
              'gpu_peak_total_used_mib': max(t.get('gpu_used_mib', 0) for t in telemetry),
              'gpu_peak_utilization_percent': max(t.get('gpu_utilization_percent', 0) for t in telemetry),
              'system_ram_min_available_mib': min((t['system_ram_available_mib'] for t in telemetry if 'system_ram_available_mib' in t),default=None),
              'gpu_measurement_scope': 'whole GPU including desktop/other processes',
              'telemetry': telemetry}
    if result['status'] == 'done':
        report.update(event_count=len(result['violations']),
                      events=[{'rule_id': event.get('rule_id'), 'timecode_sec': event['timecode_sec'],
                               'evidence_timestamps': [frame['timestamp_sec'] for frame in event['evidence_frames']]}
                              for event in result['violations']],
                      evidence_checks=measurement.check_evidence(result, args.video, load_config(args.config)))
    else:
        report['error'] = result.get('error')
        if 'verdict' in result:
            raise RuntimeError('Failed job must not contain a verdict')
    report['evidence_valid'] = all(check['pts_matches'] and check['jpeg_matches']
                                   for check in report.get('evidence_checks', [])) if report['status'] == 'done' else None
    measurement.write_measurement(output,report)
    print(json.dumps({k: v for k, v in report.items() if k != 'telemetry'}, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'done' and report['evidence_valid'] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/local.json'))
    parser.add_argument('--api-url', default='http://127.0.0.1:8001')
    args = parser.parse_args()
    output = args.output_dir
    started = time.monotonic()
    try:
        output = output.resolve()
        measurement.prepare_measurement(output)
        return run_api_measurement(args,output)
    except (Exception,KeyboardInterrupt) as error:
        return measurement.measurement_failure(output,started,error)


if __name__ == '__main__':
    raise SystemExit(main())
