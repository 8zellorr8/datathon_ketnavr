"""CPU fixtures for evaluation integrity; no model or sockets are used."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from forte_ai.evaluation import build_run_fingerprint, run_evaluation, summarize


JPEG = base64.b64encode(b'\xff\xd8fixture\xff\xd9').decode('ascii')


def prediction(verdict='violation', timestamp=57.0):
    return {
        'status': 'done', 'verdict': verdict,
        'violations': [] if verdict == 'no_violation' else [{
            'type': 'Visible event', 'description': 'Concrete visible action',
            'timecode_sec': timestamp,
            'evidence_frames': [{'timestamp_sec': timestamp, 'image_base64': JPEG}],
        }],
        'analysis': {'elapsed_sec': 2.0, 'duration_sec': 72.7},
    }


def record(identifier='X-4', label=1, event=57.0, digest='same-content'):
    return {'id': identifier, 'label': label, 'event_time_sec': event,
            'sha256': digest, 'video_path': 'never-opened.mov'}


class EvaluationIntegrityTests(unittest.TestCase):
    def test_malformed_done_results_are_failures_not_negatives(self):
        malformed = [None, [], {'status': 'processing'},
                     {'status': 'done', 'verdict': 'no_violation'},
                     {'status': 'done', 'verdict': 'violation', 'violations': []},
                     {'status': 'done', 'verdict': 'unknown', 'violations': []}]
        for field, value in [('timecode_sec', -1), ('timecode_sec', float('nan')),
                             ('timecode_sec', 90), ('description', ''),
                             ('evidence_frames', []), ('type', None)]:
            item = prediction()
            item['violations'][0][field] = value
            malformed.append(item)
        item = prediction()
        item['violations'][0]['evidence_frames'][0]['image_base64'] = 'not-jpeg'
        malformed.append(item)
        item = prediction('no_violation')
        item['analysis']['elapsed_sec'] = float('inf')
        malformed.append(item)
        for bad in malformed:
            with self.subTest(result=bad):
                report = summarize([record(label=0, event=None)], {'X-4': bad})
                self.assertEqual(report['completed'], 0)
                self.assertEqual(report['failed_or_unprocessed'], 1)
                self.assertEqual(report['confusion_matrix_completed_only']['true_negative'], 0)
                self.assertEqual(report['correct_fraction_including_failures'], 0)

    def test_original_and_acceptable_duplicate_timestamps_are_separate(self):
        rows = [record(), record('X-47', event=68.0)]
        original = copy.deepcopy(rows)
        results = {'X-4': prediction(), 'X-47': prediction()}
        report = summarize(rows, results)
        self.assertEqual(rows, original)
        self.assertEqual(report['completed'], 2)
        self.assertEqual(report['median_localization_error_sec'], 5.5)
        self.assertEqual(report['localization_ambiguity']['median_acceptable_gt_error_sec'], 0)
        conflict = report['localization_ambiguity']['conflicting_exact_content_groups'][0]
        self.assertEqual(conflict['acceptable_gt_timestamps_sec'], [57.0, 68.0])
        self.assertEqual(conflict['original_event_times_by_id'], {'X-4': 57.0, 'X-47': 68.0})
        unique = report['unique_content']
        self.assertEqual(unique['selected_videos'], 1)
        self.assertEqual(unique['confusion_matrix_completed_only']['true_positive'], 1)
        self.assertEqual(unique['median_localization_error_sec'], 0)
        self.assertEqual(unique['groups'][0]['aliases'], ['X-47'])

    def test_unique_metrics_do_not_choose_successful_alias_after_canonical_failure(self):
        rows = [record(), record('X-47', event=68.0)]
        report = summarize(rows, {'X-4': {'status': 'failed'}, 'X-47': prediction()})
        self.assertEqual(report['completed'], 1)
        self.assertEqual(report['unique_content']['completed'], 0)
        self.assertEqual(report['unique_content']['failed_ids'], ['X-4'])
        self.assertTrue(report['unique_content']['groups'][0]['prediction_disagreement'])

    def test_conflicting_duplicate_labels_are_not_resolved_by_voting(self):
        rows = [record(), record('X-47', label=0, event=None)]
        report = summarize(rows, {'X-4': prediction(), 'X-47': prediction('no_violation')})
        self.assertEqual(report['completed'], 2)
        self.assertEqual(report['unique_content']['completed'], 0)
        self.assertTrue(report['unique_content']['groups'][0]['ground_truth_label_conflict'])

    def test_rows_without_hashes_remain_separate(self):
        rows = [record(), record('X-47')]
        for row in rows:
            row.pop('sha256')
        report = summarize(rows, {'X-4': prediction(), 'X-47': prediction()})
        self.assertEqual(report['unique_content']['selected_videos'], 2)


class EvaluationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rules = self.root / 'rules.json'
        self.rules.write_text('[]', encoding='utf-8')
        self.code = self.root / 'code'
        self.code.mkdir()
        self.prompt = self.code / 'local_vlm.py'
        self.prompt.write_text("PROMPT = 'first'\n", encoding='utf-8')
        self.config = {'model': 'local-alias', 'expected_digest': 'a' * 64,
                       'rules_path': str(self.rules)}

    def fingerprint(self, config=None, **kwargs):
        return build_run_fingerprint(config or self.config, code_root=self.code, **kwargs)

    def test_fingerprint_changes_when_rules_change_at_same_path(self):
        before, _ = self.fingerprint()
        self.rules.write_text('[{"id":"new_rule"}]', encoding='utf-8')
        after, _ = self.fingerprint()
        self.assertNotEqual(before, after)

    def test_fingerprint_changes_when_prompt_code_changes_at_same_path(self):
        before, _ = self.fingerprint()
        self.prompt.write_text("PROMPT = 'second'\n", encoding='utf-8')
        after, _ = self.fingerprint()
        self.assertNotEqual(before, after)

    def test_fingerprint_records_declared_digest_without_querying_model(self):
        before, metadata = self.fingerprint()
        after, changed = self.fingerprint({**self.config, 'expected_digest': 'b' * 64})
        self.assertNotEqual(before, after)
        self.assertFalse(metadata['model_digest_verified_by_evaluator'])
        self.assertEqual(changed['model_digest_provenance'], 'config_expected_digest')
        supplied, details = self.fingerprint(run_metadata={'model_digest': 'c' * 64})
        self.assertNotEqual(before, supplied)
        self.assertEqual(details['declared_model_digest'], 'c' * 64)
        self.assertEqual(details['model_digest_provenance'], 'caller_run_metadata')

    def test_run_saves_malformed_prediction_as_failed_and_discloses_prior_runs(self):
        row = record('X-26', event=9.0)
        manifest = {'records': [row]}
        splits = {'ids': {'development': [], 'validation': ['X-26'], 'test': []}}
        malformed = {'status': 'done', 'verdict': 'no_violation'}
        output = self.root / 'results'
        with patch('forte_ai.evaluation.analyze_video', return_value=malformed) as analyze:
            report = run_evaluation(manifest, splits, self.config, output)
        analyze.assert_called_once_with('never-opened.mov', self.config)
        saved = json.loads((output / 'X-26.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['result']['status'], 'failed')
        self.assertNotIn('verdict', saved['result'])
        self.assertEqual(saved['result']['error']['type'], 'MalformedPrediction')
        self.assertEqual(report['completed'], 0)
        self.assertFalse(report['run_metadata']['blind_holdout'])
        self.assertEqual(report['run_metadata']['known_prior_inference_ids_in_partition'], ['X-26'])
        self.assertEqual(report['run_metadata']['known_prior_inference_ids'], ['X-26', 'X-46'])
        self.assertEqual(saved['config_fingerprint'], report['run_metadata']['config_fingerprint'])
        self.assertTrue(report['run_metadata']['fingerprint_components']['rules_content_sha256'])


if __name__ == '__main__':
    unittest.main()
