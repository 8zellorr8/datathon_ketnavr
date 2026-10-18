from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='ForteBank local AI prototype')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Import 42 reviewed videos and make grouped partitions')
    prepare.add_argument('--workbook', required=True)
    prepare.add_argument('--videos-dir', required=True)
    prepare.add_argument('--output', default='data')
    analyze = commands.add_parser('analyze', help='Analyze one new local .mov or .mp4 video')
    analyze.add_argument('--video', required=True)
    analyze.add_argument('--config', default='config/local.json')
    analyze.add_argument('--output', default='results/single.json')
    evaluate = commands.add_parser('evaluate', help='Benchmark on a fixed grouped partition')
    evaluate.add_argument('--manifest', default='data/manifest.json')
    evaluate.add_argument('--splits', default='data/splits.json')
    evaluate.add_argument('--config', default='config/local.json')
    evaluate.add_argument('--partition', choices=['development', 'validation', 'test'], default='validation')
    evaluate.add_argument('--limit', type=int)
    evaluate.add_argument('--output', default='results/validation')
    serve = commands.add_parser('serve', help='Start the local asynchronous jury API')
    serve.add_argument('--config', default='config/local.json')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            from .dataset import prepare_dataset
            from .evaluation import make_splits
            manifest = prepare_dataset(args.workbook, args.videos_dir, args.output)
            splits = make_splits(manifest)
            destination = Path(args.output) / 'splits.json'
            destination.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'videos': len(manifest['records']), 'partitions': splits['counts']}, ensure_ascii=False, indent=2))
        elif args.command == 'analyze':
            from .pipeline import analyze_video, load_config
            result = analyze_video(args.video, load_config(args.config))
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'status': result['status'], 'verdict': result['verdict'], 'events': len(result['violations']),
                              'output': str(destination.resolve())}, ensure_ascii=False, indent=2))
        elif args.command == 'evaluate':
            from .evaluation import run_evaluation
            from .pipeline import load_config
            report = run_evaluation(_json(args.manifest), _json(args.splits), load_config(args.config), args.output,
                                    partition=args.partition, limit=args.limit)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == 'serve':
            from .pipeline import load_config
            from .server import serve
            key = os.environ.get('FORTE_API_KEY', '')
            if len(key) < 16:
                raise ValueError('Set FORTE_API_KEY to a random secret of at least 16 characters')
            serve(load_config(args.config), host=args.host, port=args.port, api_key=key)
        return 0
    except Exception as exc:
        # A failed inference is an operational failure, never a clean video.
        print(json.dumps({'status': 'failed', 'error': type(exc).__name__, 'message': str(exc)[:1000]}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
