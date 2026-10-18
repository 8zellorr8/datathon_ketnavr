"""Mocked visual-review checks; no source videos, model, or network are used."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from forte_ai.evaluation import build_run_fingerprint
from forte_ai.pipeline import (
    AnalysisError, AnalysisErrorCode, _read_visual_checks, analyze_video, load_config,
)


DOOR_ID = 'door_left_open_while_working'


class VisualCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rules = self.root / 'rules.json'
        self.source = {'verified': True, 'url': 'https://example.invalid/verified-rule'}
        self.rules.write_text(json.dumps([{
            'id': 'verified_cash_rule', 'name': 'Verified visual cash rule', 'enabled': True,
            'source': self.source, 'visual_requirements': ['Visible cash action'],
            'counterexamples': ['Action obscured'],
        }]), encoding='utf-8')
        self.visual_path = self.root / 'visual_checks.json'
        self.check = {
            'id': DOOR_ID, 'name': 'Room door open during work', 'enabled': True,
            'visual_requirements': ['Room entrance remains visibly open while cashier works'],
            'counterexamples': ['Brief passage followed by closing the door', 'Safe door only'],
            'min_evidence_frames': 2, 'min_evidence_span_sec': 1.0,
        }
        self.write_checks([self.check])
        self.config = {'rules_path': str(self.rules), 'visual_checks_path': str(self.visual_path)}

    def write_checks(self, checks):
        self.visual_path.write_text(json.dumps(checks), encoding='utf-8')

    @staticmethod
    def frame(_path, timestamp, **_kwargs):
        return {'timestamp_sec': float(timestamp), 'image_base64': 'mock_jpeg'}

    @staticmethod
    def response(candidates=()):
        return {'observations': [], 'summary': '', 'candidate_violations': list(candidates)}

    @staticmethod
    def candidate(frames, rule_id=DOOR_ID, evidence_times=(0.0, 4.0)):
        times = [frame['timestamp_sec'] for frame in frames]
        indices = [times.index(timestamp) for timestamp in evidence_times]
        main = times.index(4.0)
        return {
            'rule_id': rule_id, 'frame_index': main, 'timecode_sec': times[main],
            'description': 'Visible room door remains open while work continues.',
            'confidence': .95, 'evidence_frame_indices': indices,
        }

    def run_mocked(self, segment, config=None):
        with patch('forte_ai.pipeline.probe_video', return_value={'duration_sec': 8.0}), \
             patch('forte_ai.pipeline.extract_frame', side_effect=self.frame), \
             patch('forte_ai.pipeline.analyze_segment', side_effect=segment):
            return analyze_video('unseen-random-upload.mov', config or self.config)

    def test_no_optional_catalogue_is_empty(self):
        self.assertEqual(_read_visual_checks(None), [])

    def test_reader_cannot_promote_observation_to_verified_source(self):
        self.write_checks([{**self.check, 'source': self.source,
                            'verification_status': 'normative_source_verified',
                            'detection_basis': 'verified_normative_rule',
                            'label': 1, 'event_time_sec': 6.0}])
        parsed = _read_visual_checks(self.visual_path)[0]
        self.assertIsNone(parsed.get('source'))
        self.assertEqual(parsed['verification_status'], 'needs_regulation_review')
        self.assertEqual(parsed['detection_basis'], 'visual_observation')
        self.assertNotIn('label', parsed)
        self.assertNotIn('event_time_sec', parsed)

    def test_disabled_visual_check_is_not_active(self):
        self.write_checks([{**self.check, 'enabled': False}])
        self.assertEqual(_read_visual_checks(self.visual_path), [])

    def test_invalid_visual_catalogues_fail(self):
        invalid = [
            {}, [None], [{**self.check, 'id': 'Invalid Door'}],
            [{**self.check, 'name': ''}], [{**self.check, 'enabled': 'true'}],
            [{**self.check, 'visual_requirements': []}],
            [{**self.check, 'counterexamples': ['']}],
            [{**self.check, 'min_evidence_frames': 1}],
            [{**self.check, 'min_evidence_span_sec': 0}],
            [self.check, self.check],
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.write_checks(value)
                with self.assertRaises(AnalysisError) as failure:
                    _read_visual_checks(self.visual_path)
                self.assertIs(failure.exception.safe_code, AnalysisErrorCode.INVALID_RULE_CATALOGUE)

    def test_load_config_resolves_optional_catalogue_relative_to_config(self):
        config_dir = self.root / 'config'
        config_dir.mkdir()
        config_path = config_dir / 'local.json'
        config_path.write_text(json.dumps({'rules_path': '../rules.json',
                                           'visual_checks_path': '../visual_checks.json'}), encoding='utf-8')
        loaded = load_config(config_path)
        self.assertEqual(loaded['visual_checks_path'], str(self.visual_path.resolve()))
        self.assertEqual(loaded['rules_path'], str(self.rules.resolve()))
        config_path.write_text('{"rules_path":"../rules.json"}', encoding='utf-8')
        self.assertIsNone(load_config(config_path)['visual_checks_path'])

    def test_confirmed_visual_signal_has_evidence_without_legal_source(self):
        result = self.run_mocked(lambda frames, **_kwargs: self.response([self.candidate(frames)]))
        self.assertEqual(result['verdict'], 'violation')
        event = result['violations'][0]
        self.assertEqual(event['rule_id'], DOOR_ID)
        self.assertEqual(event['verification_status'], 'needs_regulation_review')
        self.assertIsNone(event['rule_source'])
        self.assertIs(event['requires_human_review'], True)
        self.assertTrue(event['regulatory_note'].strip())
        self.assertEqual({frame['timestamp_sec'] for frame in event['evidence_frames']}, {0.0, 4.0})
        self.assertIn(DOOR_ID, result['analysis']['visual_check_ids'])
        self.assertNotIn(DOOR_ID, result['analysis']['verified_rule_ids'])
        self.assertEqual(result['analysis']['coverage'], 'configured_checks_only')

    def test_refinement_may_reject_door_hypothesis(self):
        calls = []
        def segment(frames, **_kwargs):
            calls.append(frames)
            return self.response([self.candidate(frames)]) if len(calls) == 1 else self.response()
        result = self.run_mocked(segment)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['verdict'], 'no_violation')
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['analysis']['coverage'], 'configured_checks_only')

    def test_single_confirmed_frame_is_insufficient_for_persistent_door_signal(self):
        calls = []
        def segment(frames, **_kwargs):
            calls.append(frames)
            times = (0.0, 4.0) if len(calls) == 1 else (4.0,)
            return self.response([self.candidate(frames, evidence_times=times)])
        self.assertEqual(self.run_mocked(segment)['verdict'], 'no_violation')

    def test_two_nearby_confirmed_frames_are_insufficient_for_required_span(self):
        calls = []
        def segment(frames, **_kwargs):
            calls.append(frames)
            times = (0.0, 4.0) if len(calls) == 1 else (4.0, 4.5)
            return self.response([self.candidate(frames, evidence_times=times)])
        self.assertEqual(self.run_mocked(segment)['verdict'], 'no_violation')

    def test_model_receives_visual_criteria_without_annotations_or_legal_claims(self):
        self.write_checks([{**self.check, 'source': self.source, 'label': 1, 'event_time_sec': 6.0}])
        calls = []
        def segment(frames, **kwargs):
            calls.append(kwargs)
            return self.response([self.candidate(frames)])
        self.run_mocked(segment)
        self.assertEqual(len(calls), 2)
        for call in calls:
            for rule in call['rules']:
                self.assertEqual(set(rule), {'rule_id', 'visual_requirements', 'counterexamples'})
            encoded = json.dumps(call)
            self.assertNotIn('event_time_sec', encoded)
            self.assertNotIn('example.invalid', encoded)
            self.assertNotIn('unseen-random-upload', encoded)
            self.assertNotIn('normative_source_verified', encoded)

    def test_mixed_events_preserve_normative_and_visual_source_separation(self):
        calls = []
        def segment(frames, **kwargs):
            calls.append(kwargs)
            identifiers = [rule['rule_id'] for rule in kwargs['rules']]
            return self.response([self.candidate(frames, identifier) for identifier in identifiers])
        result = self.run_mocked(segment)
        self.assertEqual(len(calls), 3)
        events = {event['rule_id']: event for event in result['violations']}
        self.assertEqual(set(events), {'verified_cash_rule', DOOR_ID})
        verified = events['verified_cash_rule']
        self.assertEqual(verified['verification_status'], 'normative_source_verified')
        self.assertEqual(verified['rule_source'], self.source)
        self.assertEqual(events[DOOR_ID]['verification_status'], 'needs_regulation_review')
        self.assertIsNone(events[DOOR_ID]['rule_source'])

    def test_duplicate_ids_between_catalogues_fail_before_video_or_model(self):
        self.write_checks([{**self.check, 'id': 'verified_cash_rule'}])
        with patch('forte_ai.pipeline.probe_video') as probe, \
             patch('forte_ai.pipeline.analyze_segment') as model:
            with self.assertRaises(AnalysisError) as failure:
                analyze_video('unseen-random-upload.mov', self.config)
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.INVALID_RULE_CATALOGUE)
        probe.assert_not_called()
        model.assert_not_called()


class VisualCatalogueFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rules = self.root / 'rules.json'
        self.rules.write_text('[]', encoding='utf-8')
        self.visual = self.root / 'visual.json'
        self.visual.write_text('[]', encoding='utf-8')
        self.code = self.root / 'code'
        self.code.mkdir()
        (self.code / 'pipeline.py').write_text('# frozen mock source\n', encoding='utf-8')
        self.config = {'model': 'local-vision-model', 'rules_path': str(self.rules)}

    def fingerprint(self, config):
        return build_run_fingerprint(config, code_root=self.code)

    def test_visual_catalogue_bytes_are_hashed_at_same_path(self):
        config = {**self.config, 'visual_checks_path': str(self.visual)}
        before, components = self.fingerprint(config)
        self.assertEqual(components['visual_checks_content_sha256'],
                         hashlib.sha256(self.visual.read_bytes()).hexdigest())
        self.visual.write_text('[{"id":"changed"}]', encoding='utf-8')
        after, _ = self.fingerprint(config)
        self.assertNotEqual(before, after)

    def test_none_or_missing_optional_catalogue_preserves_legacy_component_layout(self):
        for config in (self.config, {**self.config, 'visual_checks_path': None}):
            with self.subTest(config=config):
                before, components = self.fingerprint(config)
                self.assertEqual(components['fingerprint_version'], 2)
                self.assertNotIn('visual_checks_content_sha256', components)
                self.visual.write_text('[{"id":"irrelevant"}]', encoding='utf-8')
                self.assertEqual(before, self.fingerprint(config)[0])

    def test_explicit_missing_visual_catalogue_fails_instead_of_ignoring_it(self):
        with self.assertRaises(FileNotFoundError):
            self.fingerprint({**self.config, 'visual_checks_path': str(self.root / 'absent.json')})


if __name__ == '__main__':
    unittest.main()
