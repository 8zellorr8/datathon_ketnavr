"""Run one blinded local video and save GPU/RAM, timing and evidence checks.

Only the randomly named video path reaches the analyzer. Expected labels and
timestamps are not accepted. Console output contains technical metadata only.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent


def write_measurement(output: Path, report: dict) -> None:
    """Replace the report atomically so a rerun never leaves stale success."""
    temporary = output / ('.measurement-' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(output / 'measurement.json')
    finally:
        temporary.unlink(missing_ok=True)


def prepare_measurement(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_measurement(output, {'status': 'processing', 'verdict': None})
    (output / 'prediction.json').unlink(missing_ok=True)


def measurement_failure(output: Path, started: float, error: BaseException) -> int:
    """Report only a constant exception class, never exception/private text."""
    report = {'status': 'measurement_failed', 'verdict': None,
              'elapsed_sec': round(time.monotonic() - started, 3),
              'error': {'class': type(error).__name__}}
    try:
        write_measurement(output, report)
    except Exception as write_error:
        report['report_write_error'] = {'class': type(write_error).__name__}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 1


class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
              ('total_phys',ctypes.c_ulonglong),('avail_phys',ctypes.c_ulonglong),
              ('total_page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),
              ('total_virtual',ctypes.c_ulonglong),('avail_virtual',ctypes.c_ulonglong),
              ('avail_extended',ctypes.c_ulonglong)]


def snapshot(start: float) -> dict:
    row={'elapsed_sec':round(time.monotonic()-start,3)}
    memory=Memory(); memory.length=ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        row['system_ram_available_mib']=round(memory.avail_phys/2**20,1)
    try:
        value=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW)
        used,total,util=(float(v.strip()) for v in value.stdout.splitlines()[0].split(','))
        row.update(gpu_used_mib=used,gpu_total_mib=total,gpu_utilization_percent=util)
    except (OSError,subprocess.SubprocessError,ValueError,IndexError):
        row['gpu_sample_unavailable']=True
    return row


def check_evidence(result: dict, video: Path, config: dict) -> list:
    from io import BytesIO
    from PIL import Image
    from forte_ai.server import _validate_analysis_result
    from forte_ai.video import extract_frame
    _validate_analysis_result(result)
    checks=[]
    for event_index,event in enumerate(result['violations']):
        for frame in event['evidence_frames']:
            timestamp=frame['timestamp_sec']
            data=base64.b64decode(frame['image_base64'],validate=True)
            with Image.open(BytesIO(data)) as image:
                image.verify()
            repeated=extract_frame(video,max(0,timestamp-0.0001),max_side=int(config['frame_max_side']))
            checks.append({'event_index':event_index,'timestamp_sec':timestamp,
                'redecoded_timestamp_sec':repeated['timestamp_sec'],
                'pts_matches':abs(timestamp-repeated['timestamp_sec'])<=0.001,
                'jpeg_matches':hashlib.sha256(data).digest()==hashlib.sha256(base64.b64decode(repeated['image_base64'])).digest()})
    return checks


def run_analysis_measurement(args, output: Path) -> int:
    scratch=ROOT/'data/runtime'/('measure-'+uuid.uuid4().hex)
    blinded=scratch/'upload.mov'
    prediction=output/'prediction.json'
    started=time.monotonic()
    telemetry=[]
    child=None
    try:
        scratch.mkdir(parents=True)
        shutil.copyfile(args.video.resolve(),blinded)
        with (output/'stdout.log').open('w',encoding='utf-8') as stdout, (output/'stderr.log').open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen([sys.executable,'-m','forte_ai','analyze','--video',str(blinded),
                '--config',str(args.config.resolve()),'--output',str(prediction)],cwd=ROOT,
                stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            last_report=-20
            while child.poll() is None:
                telemetry.append(snapshot(started))
                elapsed=time.monotonic()-started
                if elapsed-last_report>=20:
                    print(json.dumps({'status':'processing','elapsed_sec':round(elapsed,1)}),flush=True)
                    last_report=elapsed
                if elapsed>295:
                    child.kill(); child.wait()
                    break
                time.sleep(1)
        elapsed=time.monotonic()-started
        telemetry.append(snapshot(started))
        report={'exit_code':child.returncode,'elapsed_sec':round(elapsed,3),
            'gpu_peak_total_used_mib':max((r.get('gpu_used_mib',0) for r in telemetry),default=0),
            'gpu_peak_utilization_percent':max((r.get('gpu_utilization_percent',0) for r in telemetry),default=0),
            'system_ram_min_available_mib':min((r['system_ram_available_mib'] for r in telemetry if 'system_ram_available_mib' in r),default=None),
            'gpu_measurement_scope':'whole GPU including desktop/other processes',
            'blinded_input_name':blinded.name,'telemetry':telemetry}
        if prediction.exists() and child.returncode==0:
            from forte_ai.pipeline import load_config
            result=json.loads(prediction.read_text(encoding='utf-8'))
            report.update(status=result['status'],verdict=result['verdict'],
                event_count=len(result['violations']),
                events=[{'rule_id':e.get('rule_id'),'timecode_sec':e['timecode_sec'],
                    'evidence_timestamps':[f['timestamp_sec'] for f in e['evidence_frames']]} for e in result['violations']],
                analysis=result.get('analysis'),
                evidence_checks=check_evidence(result,blinded,load_config(args.config)))
        else:
            report.update(status='failed',verdict=None,event_count=None)
            lines=(output/'stderr.log').read_text(encoding='utf-8',errors='replace').splitlines()
            for line in reversed(lines):
                try:
                    error=json.loads(line)
                    # Child exception messages can contain private paths/text.
                    error_class=error.get('error')
                    if isinstance(error_class,str) and error_class.isidentifier():
                        report['error']={'class':error_class}
                    break
                except ValueError:
                    continue
        report['evidence_valid'] = all(check['pts_matches'] and check['jpeg_matches']
                                       for check in report.get('evidence_checks', [])) if report['status']=='done' else None
        write_measurement(output,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('telemetry','analysis')},ensure_ascii=False),flush=True)
        return 0 if report['status']=='done' and report['evidence_valid'] else 1
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if blinded.resolve().is_relative_to(ROOT.resolve()):
            blinded.unlink(missing_ok=True)
            if scratch.exists():
                scratch.rmdir()


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('config/local.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir
    started=time.monotonic()
    try:
        output=output.resolve()
        prepare_measurement(output)
        return run_analysis_measurement(args,output)
    except (Exception,KeyboardInterrupt) as error:
        return measurement_failure(output,started,error)


if __name__=='__main__':
    raise SystemExit(main())
