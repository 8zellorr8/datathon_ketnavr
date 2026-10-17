"""Strict per-frame door observation contract; mocked local transport only."""
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from forte_ai import local_vlm as vlm


MODEL = 'qwen3-vl:2b-instruct-q4_K_M'
PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGncAAAAASUVORK5CYII='
FRAMES = [{'timestamp_sec': 2.0, 'image_base64': PNG},
          {'timestamp_sec': 4.5, 'image_base64': PNG}]


class Response(io.BytesIO):
    status = 200

    def __init__(self, value):
        body = json.dumps(value).encode('utf-8')
        super().__init__(body)
        self.headers = {'Content-Length': str(len(body))}


class DoorStateTests(unittest.TestCase):
    @staticmethod
    def row(index, **changes):
        return {'frame_index': index, 'door_state': 'open', 'work_state': 'working',
                'cash_visible': 'no', 'confidence': .95, **changes}

    def call(self, content=None, *, frames=None, tags=None, show=None, outer=None, **kwargs):
        self.requests = []
        if content is None:
            content = {'frames': [self.row(0), self.row(1)]}
        if tags is None:
            tags = {'models': [{'name': MODEL, 'digest': 'mock-local-digest'}]}
        if show is None:
            show = {'capabilities': ['vision']}
        if outer is None:
            outer = {'done': True, 'message': {'content': json.dumps(content)}}
        def open_request(request, timeout):
            self.requests.append(request)
            if request.full_url.endswith('/api/tags'):
                return Response(tags)
            if request.full_url.endswith('/api/show'):
                return Response(show)
            return Response(outer)
        options = {'model': MODEL, 'image_layout': 'individual', 'num_ctx': 8192, **kwargs}
        with patch.object(vlm.urllib.request, 'build_opener',
                          return_value=SimpleNamespace(open=open_request)):
            return vlm.assess_room_door(FRAMES if frames is None else frames, **options)

    def test_rows_cover_all_inputs_and_bind_source_timestamps(self):
        result = self.call({'frames': [self.row(1), self.row(0, door_state='closed', work_state='passing')]})
        self.assertEqual([row['frame_index'] for row in result['frames']], [0, 1])
        self.assertEqual([row['timestamp_sec'] for row in result['frames']], [2.0, 4.5])
        self.assertEqual(result['frames'][0]['door_state'], 'closed')
        self.assertEqual(result['frames'][0]['work_state'], 'passing')
        self.assertEqual(result['backend'], 'ollama')
        self.assertEqual(result['inference_meta']['model_task'], 'room_door_state')
        self.assertEqual(result['inference_meta']['output_token_limit'], 768)
        self.assertEqual(result['inference_meta']['model_digest'], 'mock-local-digest')

    def test_request_schema_is_short_and_requires_only_factual_states(self):
        self.call()
        payload = json.loads(self.requests[-1].data)
        self.assertEqual(payload['options'], {'temperature': 0, 'num_predict': 768, 'num_ctx': 8192})
        self.assertFalse(payload['stream'])
        self.assertFalse(payload['think'])
        rows = payload['format']['properties']['frames']
        self.assertEqual((rows['minItems'], rows['maxItems']), (2, 2))
        self.assertEqual(set(rows['items']['required']),
                         {'frame_index', 'door_state', 'work_state', 'cash_visible', 'confidence'})
        self.assertFalse(rows['items']['additionalProperties'])
        prompt = payload['messages'][0]['content']
        self.assertIn('workplace', prompt)
        self.assertNotIn('bank-cashier', prompt)
        self.assertIn('never a safe', prompt)
        self.assertIn('not_visible', prompt)
        self.assertIn('not simply entering', prompt)
        self.assertEqual(payload['messages'][-1]['images'], [PNG, PNG])

    def test_input_metadata_never_enters_model_payload(self):
        frames = [{**frame, 'filename': 'PRIVATE_FILENAME.mov', 'label': 'PRIVATE_GOLD',
                   'event_time_sec': 6.0, 'description': 'PRIVATE_ANNOTATION'} for frame in FRAMES]
        self.call(frames=frames)
        payload = self.requests[-1].data.decode('utf-8')
        for private in ('PRIVATE_FILENAME', 'PRIVATE_GOLD', 'PRIVATE_ANNOTATION', 'event_time_sec'):
            self.assertNotIn(private, payload)

    def test_missing_duplicate_or_extra_rows_are_rejected(self):
        malformed = [
            {'frames': []}, {'frames': [self.row(0)]},
            {'frames': [self.row(0), self.row(0)]},
            {'frames': [self.row(0), self.row(1), self.row(2)]},
            {'frames': [self.row(0), self.row(2)]},
            {'frames': [self.row(0), self.row(-1)]},
            {'frames': {}, 'extra': ''},
        ]
        for content in malformed:
            with self.subTest(content=content), self.assertRaises(vlm.LocalVLMOutputError):
                self.call(content)

    def test_row_extras_missing_fields_and_boolean_indices_fail(self):
        rows = [self.row(True), self.row(1.0), self.row(1, timestamp_sec=1000),
                self.row(1, legal_conclusion='violation')]
        missing = self.row(1)
        del missing['cash_visible']
        rows.append(missing)
        for row in rows:
            with self.subTest(row=row), self.assertRaises(vlm.LocalVLMOutputError):
                self.call({'frames': [self.row(0), row]})

    def test_unknown_states_or_invalid_confidence_fail(self):
        mutations = [
            {'door_state': 'locked'}, {'door_state': True}, {'door_state': []},
            {'work_state': 'sitting'}, {'work_state': False},
            {'cash_visible': 'probably'}, {'cash_visible': True},
            {'confidence': True}, {'confidence': -0.1}, {'confidence': 1.1},
            {'confidence': None}, {'confidence': '0.95'},
        ]
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(vlm.LocalVLMOutputError):
                self.call({'frames': [self.row(0), self.row(1, **changes)]})

    def test_uncertainty_and_away_states_are_valid_without_legal_verdict(self):
        result = self.call({'frames': [
            self.row(0, door_state='not_visible', work_state='away', cash_visible='uncertain', confidence=0),
            self.row(1, door_state='uncertain', work_state='uncertain', cash_visible='yes', confidence=1),
        ]})
        self.assertEqual(result['frames'][0]['door_state'], 'not_visible')
        self.assertEqual(result['frames'][1]['cash_visible'], 'yes')
        self.assertNotIn('verdict', result)
        self.assertNotIn('violations', result)

    def test_one_to_twelve_frames_are_supported(self):
        for count in (1, 12):
            frames = [{'timestamp_sec': float(index), 'image_base64': PNG} for index in range(count)]
            result = self.call({'frames': [self.row(index) for index in range(count)]}, frames=frames,
                               image_layout='contact_sheet', frames_per_sheet=6)
            self.assertEqual(len(result['frames']), count)
            self.assertEqual(result['inference_meta']['input_frame_count'], count)
            self.assertEqual(result['inference_meta']['input_image_count'], 1 if count == 1 else 2)

    def test_zero_or_more_than_twelve_frames_fail_before_transport(self):
        with patch.object(vlm.urllib.request, 'build_opener') as transport:
            for count in (0, 13):
                frames = [{'timestamp_sec': float(index), 'image_base64': PNG} for index in range(count)]
                with self.subTest(count=count), self.assertRaises(vlm.LocalVLMConfigurationError):
                    vlm.assess_room_door(frames, model=MODEL)
            transport.assert_not_called()

    def test_contact_sheet_mapping_keeps_original_frame_indices(self):
        result = self.call(image_layout='contact_sheet', frames_per_sheet=4)
        payload = json.loads(self.requests[-1].data)
        prompt = json.loads(payload['messages'][-1]['content'])
        self.assertEqual([row['frame_index'] for row in prompt['frame_map']], [0, 1])
        self.assertEqual([row['timestamp_sec'] for row in prompt['frame_map']], [2.0, 4.5])
        self.assertEqual(result['inference_meta']['input_image_count'], 1)

    def test_nonlocal_origin_and_cloud_tag_never_reach_transport(self):
        with patch.object(vlm.urllib.request, 'build_opener') as transport:
            for kwargs in ({'base_url': 'https://example.invalid'}, {'model': 'qwen-cloud'}):
                with self.subTest(kwargs=kwargs), self.assertRaises(vlm.LocalVLMConfigurationError):
                    vlm.assess_room_door(FRAMES, **{'model': MODEL, **kwargs})
            transport.assert_not_called()

    def test_cloud_alias_is_rejected_before_frames_are_posted(self):
        with self.assertRaises(vlm.LocalVLMConfigurationError):
            self.call(show={'capabilities': ['vision'], 'remote_host': 'https://example.invalid'})
        self.assertFalse(any(request.full_url.endswith('/api/chat') for request in self.requests))

    def test_incomplete_output_is_failure(self):
        outer = {'done': True, 'done_reason': 'length',
                 'message': {'content': json.dumps({'frames': [self.row(0), self.row(1)]})}}
        with self.assertRaises(vlm.LocalVLMOutputError):
            self.call(outer=outer)

    def test_context_limit_is_not_silently_increased(self):
        with patch.object(vlm.urllib.request, 'build_opener') as transport:
            with self.assertRaises(vlm.LocalVLMConfigurationError):
                vlm.assess_room_door(FRAMES, model=MODEL, image_layout='individual', num_ctx=2048)
            transport.assert_not_called()


if __name__ == '__main__':
    unittest.main()
