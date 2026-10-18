"""Relocate bundled dataset paths without changing labels or partition membership."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

TRAINING_WARNING = (
    'Перед обучением отдельно обновите пути и provenance в training/review и export. '
    'Запускатель не изменял подтверждённые обучающие ответы.'
)


def rebase_manifest(root: Path, runtime_dir: Path) -> dict:
    """Rebase only a manifest whose bundled video bytes match every saved SHA-256."""
    def answer(status: str, reason: str, **extra) -> dict:
        return {'status': status, 'reason': reason,
                'training_warning': TRAINING_WARNING, **extra}

    root = Path(root).resolve()
    runtime_dir = Path(runtime_dir).resolve()
    manifest_path = root / 'data' / 'manifest.json'
    if not manifest_path.is_file():
        return answer('absent', 'manifest_not_included_ui_uploads_still_available')
    temporary = None
    try:
        original = manifest_path.read_bytes()
        manifest = json.loads(original.decode('utf-8-sig'))
        records = manifest.get('records') if isinstance(manifest, dict) else None
        if not isinstance(records, list) or not records:
            return answer('deferred', 'invalid_manifest_records')
        videos = (root / 'data' / 'videos').resolve()
        if not videos.is_relative_to(root) or not runtime_dir.is_relative_to(root):
            return answer('deferred', 'unsafe_dataset_or_runtime_directory')
        changed = False
        checked = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get('video_path'), str):
                return answer('deferred', 'invalid_video_path')
            leaf = record['video_path'].replace('\\', '/').split('/')[-1]
            if not leaf or leaf in ('.', '..'):
                return answer('deferred', 'invalid_video_filename')
            target = (videos / leaf).resolve()
            if not target.is_relative_to(videos) or not target.is_file():
                return answer('deferred', 'bundled_video_missing_or_outside_dataset')
            expected = record.get('sha256')
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
                return answer('deferred', 'missing_or_invalid_video_sha256')
            new_path = str(target)
            changed = changed or record['video_path'] != new_path
            record['video_path'] = new_path
            checked.append((target, expected.lower()))
        if not changed:
            return answer('unchanged', 'video_paths_already_current', records=len(records))
        for field, leaf in (('source_workbook', 'labels_full.xlsx'),
                            ('audit_path', 'dataset_audit.json')):
            target = (root / 'data' / leaf).resolve()
            if target.is_file() and target.is_relative_to(root):
                changed = changed or manifest.get(field) != str(target)
                manifest[field] = str(target)
        if not changed:
            return answer('unchanged', 'paths_already_current', records=len(records))
        for target, expected in checked:
            digest = hashlib.sha256()
            with target.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                return answer('deferred', 'video_bytes_differ_from_manifest')
        payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        backups = runtime_dir / 'portable' / 'backups'
        backups.mkdir(parents=True, exist_ok=True)
        backup = backups / ('manifest-' + uuid.uuid4().hex + '.json')
        with backup.open('xb') as destination:
            destination.write(original)
        descriptor, temporary = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp',
                                                dir=manifest_path.parent)
        with os.fdopen(descriptor, 'wb') as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
        return answer('rebased', 'video_hashes_verified_splits_preserved',
                      records=len(records), backup=str(backup))
    except (OSError, ValueError, TypeError, UnicodeError):
        return answer('deferred', 'manifest_relocation_failed')
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
