"""Scene-state orchestration with synthetic frames and mocked local models."""

import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from forte_ai import pipeline
from forte_ai.local_vlm import LocalVLMOutputError, LocalVLMTimeoutError


CASH_IDS = {
    'unattended_cash', 'cash_not_secured_between_clients', 'count_after_client_exit',
    'cash_issued_without_count', 'uncounted_unsealed_cash',
}
NORMATIVE_IDS = CASH_IDS | {'personal_items_on_desk'}
DOOR_ID = 'door_left_open_while_working'
DOOR_RULE = {
    'id': DOOR_ID, 'rule_id': DOOR_ID, 'name': 'Synthetic open-door observation',
    'source': None, 'verification_status': 'needs_regulation_review',
    'detection_basis': 'visual_observation', 'visual_requirements': [],
    'counterexamples': [], 'min_evidence_frames': 2, 'min_evidence_span_sec': 1.0,
}


def synthetic_frame(_path, timestamp, **_kwargs):
    return {'timestamp_sec': float(timestamp), 'image_base64': 'synthetic-no-real-image'}


def state(index, *, door='closed', work='working', cash='no', confidence=.95):
    return {
        'frame_index': index, 'door_state': door, 'work_state': work,
        'cash_visible': cash, 'confidence': confidence,
    }


def states_response(frames, **values):
    return {'frames': [state(index, **values) for index in range(len(frames))]}


def empty_segment(_frames, **_kwargs):
    return {'observations': [], 'summary': 'Synthetic scene only.', 'candidate_violations': []}


class ScenePipelineTests(unittest.TestCase):
    def analyze(self, scene=None, *, segment=None, config=None):
        normative_rules = [{
            'id': identifier, 'rule_id': identifier, 'name': identifier,
            'enabled': True, 'source': {'verified': True, 'title': 'Synthetic public source'},
            'visual_requirements': [], 'counterexamples': [],
        } for identifier in sorted(NORMATIVE_IDS)]
        settings = {'rules_path': 'never-read-synthetic-rules.json',
                    **({'scene_state_scan': True} if config is None else config)}
        with ExitStack() as stack:
            stack.enter_context(patch('forte_ai.pipeline._read_rules', return_value=normative_rules))
            stack.enter_context(patch('forte_ai.pipeline._read_visual_checks',
                                      return_value=[copy.deepcopy(DOOR_RULE)]))
            stack.enter_context(patch('forte_ai.pipeline.probe_video',
                                      return_value={'duration_sec': 12.15}))
            stack.enter_context(patch('forte_ai.pipeline.extract_frame', side_effect=synthetic_frame))
            scene_mock = stack.enter_context(patch(
                'forte_ai.pipeline.assess_room_door',
                side_effect=scene or (lambda frames, **_kwargs: states_response(frames))))
            segment_mock = stack.enter_context(patch(
                'forte_ai.pipeline.analyze_segment', side_effect=segment or empty_segment))
            result = pipeline.analyze_video('random-unseen-upload.mov', settings)
        return result, scene_mock, segment_mock

    def test_entry_open_during_passage_then_closed_work_is_negative(self):
        def entering(frames, **_kwargs):
            return {'frames': [
                state(index, door='open' if index == 0 else 'closed',
                      work='passing' if index == 0 else 'working')
                for index in range(len(frames))
            ]}

        result, scene, _segment = self.analyze(entering)
        self.assertEqual(result['verdict'], 'no_violation')
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['analysis']['candidate_count'], 0)
        self.assertEqual(scene.call_count, 1)

    def test_persistent_open_work_produces_a_refined_visual_event(self):
        def open_work(frames, **_kwargs):
            return states_response(frames, door='open', work='working')

        result, scene, segment = self.analyze(open_work)
        self.assertEqual(result['verdict'], 'violation')
        self.assertEqual(len(result['violations']), 1)
        event = result['violations'][0]
        self.assertEqual(event['rule_id'], DOOR_ID)
        self.assertEqual(event['verification_status'], 'needs_regulation_review')
        self.assertEqual(event['detection_basis'], 'visual_observation')
        self.assertIsNone(event['rule_source'])
        self.assertIs(event['requires_human_review'], True)
        self.assertEqual(len(event['evidence_frames']), 2)
        times = sorted(frame['timestamp_sec'] for frame in event['evidence_frames'])
        self.assertGreaterEqual(times[-1] - times[0], 1.0)
        self.assertEqual(scene.call_count, 2)
        self.assertEqual(segment.call_count, 1)
        self.assertEqual(result['analysis']['refinement_count'], 1)
        self.assertEqual([request['phase'] for request in result['analysis']['model_requests']],
                         ['scan_scene', 'scan', 'refine_scene'])

    def test_two_persistent_open_work_frames_form_a_candidate(self):
        frames = [synthetic_frame(None, timestamp) for timestamp in (4.0, 8.0)]
        candidates = pipeline._scene_candidates(
            [state(0, door='open'), state(1, door='open')], frames, DOOR_RULE, .8)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['evidence_frame_indices'], [0, 1])
        self.assertEqual(candidates[0]['timecode_sec'], 4.0)

    def test_closed_uncertain_and_transit_states_break_open_work_runs(self):
        frames = [synthetic_frame(None, timestamp) for timestamp in (0.0, 4.0, 8.0)]
        for values in ({'door': 'closed'}, {'door': 'uncertain'}, {'door': 'not_visible'},
                       {'door': 'open', 'work': 'passing'}, {'door': 'open', 'work': 'uncertain'},
                       {'door': 'open', 'confidence': .79}):
            with self.subTest(values=values):
                candidates = pipeline._scene_candidates(
                    [state(0, door='open'), state(1, **values), state(2, door='open')],
                    frames, DOOR_RULE, .8)
                self.assertEqual(candidates, [])

    def test_two_open_work_frames_under_one_second_are_insufficient(self):
        frames = [synthetic_frame(None, timestamp) for timestamp in (6.0, 6.99)]
        self.assertEqual(pipeline._scene_candidates(
            [state(0, door='open'), state(1, door='open')], frames, DOOR_RULE, .8), [])

    def test_cash_gate_excludes_exactly_five_cash_rules_and_keeps_personal_items(self):
        result, _scene, segment = self.analyze()
        self.assertEqual(set(pipeline._CASH_RULE_IDS), CASH_IDS)
        self.assertEqual({rule['rule_id'] for rule in segment.call_args.kwargs['rules']},
                         {'personal_items_on_desk'})
        self.assertEqual(result['analysis']['cash_prerequisite_exclusions'], 5)
        self.assertEqual(result['analysis']['scene_state_scan'], True)

    def test_cash_gate_accepts_the_point_nine_confidence_boundary(self):
        def clear_no_cash(frames, **_kwargs):
            return states_response(frames, cash='no', confidence=.9)

        result, _scene, segment = self.analyze(clear_no_cash)
        self.assertEqual({rule['rule_id'] for rule in segment.call_args.kwargs['rules']},
                         {'personal_items_on_desk'})
        self.assertEqual(result['analysis']['cash_prerequisite_exclusions'], 5)

    def test_any_cash_yes_uncertain_or_low_confidence_keeps_all_normative_rules(self):
        for cash, confidence in (('yes', .95), ('uncertain', .95), ('no', .899)):
            with self.subTest(cash=cash, confidence=confidence):
                def ambiguous(frames, **_kwargs):
                    rows = states_response(frames)['frames']
                    rows[-1]['cash_visible'] = cash
                    rows[-1]['confidence'] = confidence
                    return {'frames': rows}

                result, _scene, segment = self.analyze(ambiguous)
                self.assertEqual({rule['rule_id'] for rule in segment.call_args.kwargs['rules']},
                                 NORMATIVE_IDS)
                self.assertEqual(result['analysis']['cash_prerequisite_exclusions'], 0)

    def test_fine_scene_pass_can_reject_a_coarse_door_candidate(self):
        calls = []

        def coarse_open_then_closed(frames, **_kwargs):
            calls.append(len(frames))
            return states_response(frames, door='open' if len(calls) == 1 else 'closed')

        result, scene, _segment = self.analyze(coarse_open_then_closed)
        self.assertEqual(scene.call_count, 2)
        self.assertEqual(result['analysis']['candidate_count'], 1)
        self.assertEqual(result['analysis']['refinement_count'], 1)
        self.assertTrue(result['analysis']['refinements_complete'])
        self.assertEqual(result['verdict'], 'no_violation')
        self.assertEqual(result['violations'], [])

    def test_scene_timeout_propagates_without_a_clean_verdict_or_retry(self):
        calls = []

        def timed_out(*_args, **_kwargs):
            calls.append(True)
            raise LocalVLMTimeoutError('Synthetic scene timeout')

        with self.assertRaises(LocalVLMTimeoutError):
            self.analyze(timed_out)
        self.assertEqual(len(calls), 1)

    def test_repeated_malformed_scene_output_fails_after_one_retry(self):
        calls = []

        def malformed(*_args, **_kwargs):
            calls.append(True)
            raise LocalVLMOutputError('Synthetic malformed scene output')

        with self.assertRaises(LocalVLMOutputError):
            self.analyze(malformed)
        self.assertEqual(len(calls), 2)

    def test_scene_output_retry_is_recorded_when_second_attempt_succeeds(self):
        calls = []

        def retry_once(frames, **_kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise LocalVLMOutputError('Synthetic malformed first attempt')
            return states_response(frames)

        result, scene, _segment = self.analyze(retry_once)
        self.assertEqual(scene.call_count, 2)
        self.assertEqual(result['analysis']['model_calls'], 3)
        self.assertEqual(result['analysis']['model_requests'][0]['attempt_count'], 2)

    def test_explicitly_disabled_or_default_scene_scan_preserves_old_segment_path(self):
        for settings in ({'scene_state_scan': False}, {}):
            with self.subTest(settings=settings):
                self.assertIs(pipeline.DEFAULTS['scene_state_scan'], False)
                result, scene, segment = self.analyze(config=settings)
                scene.assert_not_called()
                self.assertEqual(segment.call_count, 1)
                self.assertEqual({rule['rule_id'] for rule in segment.call_args.kwargs['rules']},
                                 NORMATIVE_IDS | {DOOR_ID})
                self.assertEqual(result['analysis']['cash_prerequisite_exclusions'], 0)
                self.assertFalse(result['analysis']['scene_state_scan'])
                self.assertEqual(result['analysis']['model_calls'], 1)


if __name__ == '__main__':
    unittest.main()
