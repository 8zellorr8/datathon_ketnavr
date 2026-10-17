"""Long-video orchestration regressions; all frames and inference are mocked."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from forte_ai.pipeline import AnalysisError, AnalysisErrorCode, analyze_video
from forte_ai.local_vlm import LocalVLMOutputError, LocalVLMOutOfMemoryError, LocalVLMTimeoutError
from forte_ai.video import sample_times


class LongVideoTests(unittest.TestCase):
    DURATION = 142.53

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        rules = Path(temporary.name) / 'rules.json'
        rules.write_text(json.dumps([{
            'id': 'visible_event', 'name': 'Visible event', 'enabled': True,
            'source': {'verified': True, 'url': 'https://example.invalid/rule'},
        }]), encoding='utf-8')
        self.config = {'rules_path': str(rules), 'max_refinements': None}

    @staticmethod
    def frame(_path, timestamp, **_kwargs):
        return {'timestamp_sec': timestamp, 'image_base64': 'mock-image'}

    @staticmethod
    def response(candidates=()):
        return {'observations': [], 'summary': 'continue_scan',
                'candidate_violations': list(candidates)}

    @staticmethod
    def candidate(frames, index=0, evidence=None):
        return {
            'rule_id': 'visible_event', 'frame_index': index,
            'timecode_sec': frames[index]['timestamp_sec'],
            'description': 'Visible event', 'confidence': .95,
            'evidence_frame_indices': [index] if evidence is None else evidence,
        }

    def analyze(self, segment, *, config=None, frame=None, duration=None):
        settings = {**self.config, **(config or {})}
        with patch('forte_ai.pipeline.probe_video', return_value={
            'duration_sec': self.DURATION if duration is None else duration,
        }), patch('forte_ai.pipeline.extract_frame', side_effect=frame or self.frame), \
             patch('forte_ai.pipeline.analyze_segment', side_effect=segment):
            return analyze_video('previously-unseen-upload.mov', settings)

    def test_long_scan_covers_end_and_preserves_absolute_chronology(self):
        windows = []

        def segment(frames, **_kwargs):
            windows.append([frame['timestamp_sec'] for frame in frames])
            return self.response()

        result = self.analyze(segment)
        expected = sample_times(self.DURATION, 4)
        self.assertEqual(sorted({t for window in windows for t in window}), expected)
        self.assertGreater(windows[-1][-1], 140)
        self.assertTrue(all(window == sorted(window) for window in windows))
        self.assertEqual([window[0] for window in windows], sorted(window[0] for window in windows))
        self.assertEqual(result['analysis']['sampled_frames'], len(expected))
        self.assertEqual(result['analysis']['duration_sec'], self.DURATION)

    def test_first_model_call_precedes_full_decode_and_overlap_is_reused(self):
        decoded = []
        first_call_decoded = []

        def frame(path, timestamp, **kwargs):
            decoded.append(timestamp)
            return self.frame(path, timestamp, **kwargs)

        def segment(_frames, **_kwargs):
            if not first_call_decoded:
                first_call_decoded.append(len(decoded))
            return self.response()

        self.analyze(segment, frame=frame)
        expected = sample_times(self.DURATION, 4)
        self.assertLess(first_call_decoded[0], len(expected))
        self.assertEqual(decoded, expected)

    def test_long_window_profile_keeps_overlap_and_short_video_defaults(self):
        settings = {
            'frames_per_window': 6, 'overlap_frames': 1,
            'long_video_threshold_sec': 60,
            'long_video_frames_per_window': 12, 'long_video_overlap_frames': 2,
        }
        for duration, size, overlap in ((60, 6, 1), (60.01, 12, 2), (self.DURATION, 12, 2)):
            with self.subTest(duration=duration):
                windows = []

                def segment(frames, **_kwargs):
                    windows.append([item['timestamp_sec'] for item in frames])
                    return self.response()

                result = self.analyze(segment, config=settings, duration=duration)
                self.assertEqual(len(windows[0]), size)
                self.assertEqual(windows[1][:overlap], windows[0][-overlap:])
                self.assertEqual(sorted({t for window in windows for t in window}), sample_times(duration, 4))
                self.assertEqual(result['analysis']['effective_frames_per_window'], size)
                self.assertEqual(result['analysis']['effective_overlap_frames'], overlap)
                self.assertLessEqual(result['analysis']['peak_buffer_frames'], size)

    def candidate_scan(self):
        """Return four separated scan candidates, then reject each on review."""
        state = {'scan': 0, 'refine': 0}

        def segment(frames, **kwargs):
            if kwargs['context'] in ('', 'continue_scan'):
                state['scan'] += 1
                return self.response([self.candidate(frames)] if state['scan'] <= 4 else [])
            state['refine'] += 1
            return self.response()

        return state, segment

    def test_unlimited_candidate_review_does_not_drop_fourth_event(self):
        state, segment = self.candidate_scan()
        result = self.analyze(segment, config={'frames_per_window': 6, 'overlap_frames': 1})
        self.assertGreaterEqual(state['scan'], 4)
        self.assertEqual(state['refine'], 4)
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['verdict'], 'no_violation')

    def test_explicit_candidate_limit_still_fails_closed(self):
        _state, segment = self.candidate_scan()
        with self.assertRaises(AnalysisError) as failure:
            self.analyze(segment, config={'frames_per_window': 6, 'overlap_frames': 1,
                                          'max_refinements': 3})
        self.assertIs(failure.exception.safe_code, AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED)

    def test_model_timeout_is_not_a_clean_result(self):
        def segment(_frames, **_kwargs):
            raise TimeoutError('mock local inference timeout')

        with self.assertRaises((TimeoutError, AnalysisError)):
            self.analyze(segment)

    def test_decode_memory_failure_is_not_a_clean_result(self):
        def frame(_path, _timestamp, **_kwargs):
            raise MemoryError('mock frame allocation failure')

        with self.assertRaises((MemoryError, AnalysisError)):
            self.analyze(lambda _frames, **_kwargs: self.response(), frame=frame)

    def test_malformed_output_is_retried_once_with_same_source_frames(self):
        inputs = []

        def segment(frames, **kwargs):
            inputs.append(([frame['timestamp_sec'] for frame in frames], kwargs['context']))
            if len(inputs) == 1:
                raise LocalVLMOutputError('Invalid evidence indices')
            return self.response()

        result = self.analyze(segment, duration=8)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0][0], inputs[1][0])
        self.assertIn('Повторная техническая проверка', inputs[1][1])
        self.assertEqual(result['analysis']['model_calls'], 2)
        self.assertEqual(result['analysis']['model_requests'][0]['attempt_count'], 2)

    def test_repeated_malformed_output_remains_failure(self):
        attempts = []

        def segment(_frames, **_kwargs):
            attempts.append(1)
            raise LocalVLMOutputError('Invalid evidence indices')

        with self.assertRaises(LocalVLMOutputError):
            self.analyze(segment, duration=8)
        self.assertEqual(len(attempts), 2)

    def test_resource_errors_are_not_retried(self):
        for error_type in (LocalVLMOutOfMemoryError, LocalVLMTimeoutError):
            with self.subTest(error_type=error_type.__name__):
                attempts = []

                def segment(_frames, **_kwargs):
                    attempts.append(1)
                    raise error_type('Resource failure')

                with self.assertRaises(error_type):
                    self.analyze(segment, duration=8)
                self.assertEqual(len(attempts), 1)

    def test_refinement_preserves_causal_source_pts_and_event_time(self):
        calls = {'scan': 0, 'refine': 0}
        refined_times = []

        def frame(_path, timestamp, **_kwargs):
            # A stable 25 FPS stream whose first frame PTS is 0.02 seconds.
            pts = round(.02 + max(0, math.ceil((timestamp - .02 - 1e-8) / .04)) * .04, 6)
            return {'timestamp_sec': pts, 'image_base64': f'mock-{pts}'}

        def segment(frames, **kwargs):
            if kwargs['context'] in ('', 'continue_scan'):
                calls['scan'] += 1
                candidates = [self.candidate(frames, index=1, evidence=[0, 1])] if calls['scan'] == 1 else []
                return self.response(candidates)
            calls['refine'] += 1
            timestamps = [item['timestamp_sec'] for item in frames]
            refined_times.extend(timestamps)
            event_index = timestamps.index(4.02)
            return self.response([self.candidate(frames, index=event_index,
                                                  evidence=[timestamps.index(.02), event_index])])

        result = self.analyze(segment, frame=frame)
        self.assertEqual(calls['refine'], 1)
        self.assertIn(.02, refined_times)
        self.assertEqual(result['verdict'], 'violation')
        event = result['violations'][0]
        self.assertEqual(event['timecode_sec'], 4.02)
        self.assertEqual(event['evidence_frames'][0]['timestamp_sec'], 4.02)
        self.assertIn(.02, [item['timestamp_sec'] for item in event['evidence_frames']])


if __name__ == '__main__':
    unittest.main()
