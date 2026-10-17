import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from forte_ai.pipeline import analyze_video, AnalysisError, AnalysisErrorCode, _windows
from forte_ai.video import sample_times, VideoError
from forte_ai.evaluation import summarize, make_splits


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rules = Path(self.temp.name) / 'rules.json'
        self.rules.write_text(json.dumps([{'id': 'cash_rule', 'name': 'Проверяемый признак', 'enabled': True,
                                         'source': {'verified': True, 'url': 'https://example.invalid/norm'}}]), encoding='utf-8')
        self.config = {'rules_path': str(self.rules)}

    @staticmethod
    def frame(_path, t, **_kw):
        return {'timestamp_sec': t, 'image_base64': 'fake_jpeg_for_transport_test'}

    def run_analysis(self, responses):
        with patch('forte_ai.pipeline.probe_video', return_value={'duration_sec': 8}), \
             patch('forte_ai.pipeline.extract_frame', side_effect=self.frame), \
             patch('forte_ai.pipeline.analyze_segment', side_effect=responses):
            return analyze_video('new-unseen-name.mov', self.config)

    def test_no_candidates(self):
        response = {'observations': [], 'summary': '', 'candidate_violations': []}
        self.assertEqual(self.run_analysis([response])['verdict'], 'no_violation')

    def test_confirmed_event_contains_real_event_frame(self):
        coarse = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4, 'description': 'Видимое действие',
                  'confidence': .95, 'evidence_frame_indices': [1]}
        refined = {**coarse, 'frame_index': 5, 'timecode_sec': 4.5, 'evidence_frame_indices': [0, 5, 8]}
        responses = [dict(observations=[], summary='', candidate_violations=[coarse]),
                     dict(observations=[], summary='', candidate_violations=[refined])]
        result = self.run_analysis(responses)
        self.assertEqual(result['verdict'], 'violation')
        event = result['violations'][0]
        self.assertEqual(event['timecode_sec'], event['evidence_frames'][0]['timestamp_sec'])
        self.assertEqual(len(event['evidence_frames']), 3)
        self.assertEqual({frame['timestamp_sec'] for frame in event['evidence_frames']}, {2., 4.5, 6.})

    def test_oversized_confirmed_evidence_is_error_not_truncation_or_clean(self):
        coarse = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4,
                  'description': 'Visible hypothesis', 'confidence': .95, 'evidence_frame_indices': [1]}
        refined = {**coarse, 'frame_index': 5, 'timecode_sec': 4.5, 'evidence_frame_indices': [0, 2, 5, 8]}
        responses = [dict(observations=[], summary='', candidate_violations=[coarse]),
                     dict(observations=[], summary='', candidate_violations=[refined])]
        with self.assertRaisesRegex(AnalysisError, 'silently truncated') as failure:
            self.run_analysis(responses)
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED)

    def test_candidate_budget_failure_has_safe_code(self):
        self.config['max_refinements'] = 0
        candidate = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4,
                     'description': 'PRIVATE_MODEL_DESCRIPTION', 'confidence': .95, 'evidence_frame_indices': [1]}
        response = dict(observations=[], summary='', candidate_violations=[candidate])
        with self.assertRaises(AnalysisError) as failure:
            self.run_analysis([response])
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED)
        self.assertNotIn('PRIVATE_MODEL_DESCRIPTION', str(failure.exception))

    def test_refinement_frame_budget_failure_has_safe_code(self):
        self.config['refine_step_sec'] = .1
        candidate = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4,
                     'description': 'PRIVATE_MODEL_DESCRIPTION', 'confidence': .95, 'evidence_frame_indices': [1]}
        response = dict(observations=[], summary='', candidate_violations=[candidate])
        with self.assertRaises(AnalysisError) as failure:
            self.run_analysis([response])
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED)
        self.assertNotIn('PRIVATE_MODEL_DESCRIPTION', str(failure.exception))

    def test_analysis_error_rejects_arbitrary_diagnostic_codes(self):
        for unsafe in ('PRIVATE_MODEL_DESCRIPTION', {}, None):
            with self.subTest(value=unsafe):
                failure = AnalysisError('PRIVATE_MESSAGE', safe_code=unsafe)
                self.assertIs(failure.safe_code, AnalysisErrorCode.ANALYSIS_FAILED)
        known = AnalysisError('PRIVATE_MESSAGE', safe_code='candidate_budget_exceeded')
        self.assertIs(known.safe_code, AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED)

    def test_refinement_keeps_causal_source_frames(self):
        candidate = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4, 'description': 'Действие',
                     'confidence': .9, 'evidence_frame_indices': [0, 1]}
        calls = []
        def segment(frames, **kwargs):
            calls.append(frames)
            return {'observations': [], 'summary': '', 'candidate_violations': [candidate] if len(calls) == 1 else []}
        with patch('forte_ai.pipeline.probe_video', return_value={'duration_sec': 8}), \
             patch('forte_ai.pipeline.extract_frame', side_effect=self.frame), \
             patch('forte_ai.pipeline.analyze_segment', side_effect=segment):
            analyze_video('new.mov', self.config)
        self.assertIn(0, [f['timestamp_sec'] for f in calls[1]])

    def test_scan_and_refinement_receive_explicit_packing_and_context(self):
        candidate = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4,
                     'description': 'Visible hypothesis', 'confidence': .9, 'evidence_frame_indices': [0, 1]}
        calls = []
        def segment(frames, **kwargs):
            calls.append((frames, kwargs))
            return {'observations': [], 'summary': '', 'candidate_violations': [candidate] if len(calls) == 1 else [],
                    'inference_meta': {'input_frame_count': len(frames), 'input_image_count': 1}}
        with patch('forte_ai.pipeline.probe_video', return_value={'duration_sec': 8}), \
             patch('forte_ai.pipeline.extract_frame', side_effect=self.frame), \
             patch('forte_ai.pipeline.analyze_segment', side_effect=segment):
            result = analyze_video('new.mov', {**self.config, 'num_ctx': 8192, 'image_layout': 'contact_sheet', 'frames_per_sheet': 4})
        self.assertEqual(len(calls), 2)
        for frames, kwargs in calls:
            self.assertEqual(kwargs['num_ctx'], 8192)
            self.assertEqual(kwargs['image_layout'], 'contact_sheet')
            self.assertEqual(kwargs['frames_per_sheet'], 4)
            self.assertEqual(kwargs['contact_sheet_max_side'], 1250)
        self.assertIn(0, [frame['timestamp_sec'] for frame in calls[1][0]])
        self.assertEqual([stats['phase'] for stats in result['analysis']['model_requests']], ['scan', 'refine'])

    def test_unconfirmed_candidate_is_removed(self):
        candidate = {'rule_id': 'cash_rule', 'frame_index': 1, 'timecode_sec': 4, 'description': 'Гипотеза',
                     'confidence': .9, 'evidence_frame_indices': [1]}
        r = lambda c: dict(observations=[], summary='', candidate_violations=c)
        self.assertEqual(self.run_analysis([r([candidate]), r([])])['verdict'], 'no_violation')

    def test_backend_failure_is_not_clean_result(self):
        with self.assertRaises(RuntimeError):
            self.run_analysis([RuntimeError('backend unavailable')])

    def test_no_verified_rules_is_failure(self):
        self.rules.write_text('[]')
        with self.assertRaises(AnalysisError) as failure:
            analyze_video('anything.mov', self.config)
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.NO_VERIFIED_RULES)

    def test_expired_deadline(self):
        with self.assertRaises(AnalysisError) as failure:
            analyze_video('anything.mov', self.config, deadline=time.monotonic()-1)
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.TIME_BUDGET_EXCEEDED)

    def test_windows_no_duplicate_tail(self):
        self.assertEqual([len(w) for w in _windows(list(range(6)), 6, 1)], [6])

    def test_sampling_boundaries(self):
        self.assertEqual(sample_times(1, 4), [0])
        self.assertTrue(all(t < 10 for t in sample_times(10, 4)))
        with self.assertRaises(ValueError):
            sample_times(10, 0)

    def test_failures_are_reported_separately(self):
        rows = [{'id': 'X-1', 'label': 1, 'event_time_sec': 5}, {'id': 'X-2', 'label': 0, 'event_time_sec': None}]
        report = summarize(rows, {'X-1': {'status': 'failed'}, 'X-2': {'status': 'done', 'verdict': 'no_violation', 'violations': []}})
        self.assertEqual(report['failed_or_unprocessed'], 1)
        self.assertEqual(report['accuracy_completed_only'], 1)
        self.assertEqual(report['correct_fraction_including_failures'], .5)

    def test_grouped_split_has_no_overlap(self):
        records = [{'id': f'X-{2*g+l}', 'label': l, 'split_group': str(g)} for g in range(10) for l in (0, 1)]
        split = make_splits({'records': records})
        groups = [set(split['groups'][name]) for name in ('development', 'validation', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])


if __name__ == '__main__':
    unittest.main()
