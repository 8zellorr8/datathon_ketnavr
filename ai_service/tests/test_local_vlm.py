"""Contract and loopback transport tests; no model weights or external network."""

import errno
import base64
import io
import json
import os
import threading
import time
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from forte_ai import local_vlm as vlm


MODEL = "qwen3-vl:2b-instruct-q4_K_M"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGncAAAAASUVORK5CYII="
FRAMES = [{"timestamp_sec": 5.0, "image_base64": PNG}, {"timestamp_sec": 7.0, "image_base64": PNG}]
EVENT = {"frame_index": 0, "timestamp_sec": 5.0, "event": "cash_on_desk", "confidence": 0.8, "description": "На столе лежат купюры."}
RULES = [{"id": "safe_left_open", "name": "Открытый сейф", "description": "Сейф открыт после ухода кассира.",
          "visual_requirements": ["Сейф виден открытым", "Кассир ушёл"], "enabled": True,
          "source": {"url": "https://example.invalid/legal"}}]
CANDIDATE = {"rule_id": "safe_left_open", "frame_index": 1, "timecode_sec": 7.0,
             "description": "Кассир ушёл, дверь сейфа видна открытой.", "confidence": 0.8,
             "evidence_frame_indices": [0, 1]}


class Response(io.BytesIO):
    status = 200

    def __init__(self, content, headers=None):
        body = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
        super().__init__(body)
        self.headers = headers or {"Content-Length": str(len(body))}


class AdapterTests(unittest.TestCase):
    def call(self, content=None, *, tags=None, show=None, outer=None, frames=None, segment=False, rules=None, **kwargs):
        self.requests = []
        if tags is None:
            tags = {"models": [{"name": MODEL}]}
        if show is None:
            show = {'capabilities': ['completion', 'vision']}
        if outer is None:
            outer = {"done": True, "message": {"content": json.dumps(content if content is not None else {"events": [EVENT]})}}

        def open_request(request, timeout):
            self.requests.append((request, timeout))
            if request.full_url.endswith('/api/tags'):
                return Response(tags)
            if request.full_url.endswith('/api/show'):
                return Response(show)
            return Response(outer)

        with patch.object(vlm.urllib.request, "build_opener", return_value=SimpleNamespace(open=open_request)):
            if segment:
                return vlm.analyze_segment(frames or FRAMES, model=MODEL, rules=rules or RULES, **kwargs)
            return vlm.analyze_frames(frames or FRAMES, model=MODEL, **kwargs)

    def test_observations_request_and_validated_result(self):
        result = self.call(base_url="http://localhost:11434", num_ctx=8192)
        self.assertEqual(result["events"][0]["timestamp_sec"], 5.0)
        self.assertEqual(result["events"][0]["event"], "cash_on_desk")
        self.assertNotIn("frame_index", result["events"][0])
        self.assertEqual(result["backend"], "ollama")
        self.assertEqual(self.requests[0][0].get_method(), "GET")
        self.assertEqual(self.requests[1][0].full_url, "http://127.0.0.1:11434/api/show")
        self.assertEqual(self.requests[2][0].full_url, "http://127.0.0.1:11434/api/chat")
        request = self.requests[2][0]
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data)
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"], {"temperature": 0, "num_predict": 1000, "num_ctx": 8192})
        self.assertEqual(payload["messages"][1]["images"], [PNG, PNG])
        self.assertIn("format", payload)

    def test_empty_observations_are_valid(self):
        self.assertEqual(self.call({"events": []})["events"], [])

    def test_nonlocal_and_ambiguous_urls_never_reach_transport(self):
        urls = ["https://ollama.com", "http://192.168.1.5:11434", "http://127.0.0.2", "http://127.1",
                "http://2130706433", "http://0x7f000001", "http://localhost.evil.invalid", "http://localhost.",
                "file:///tmp/x", "http://user@localhost", "http://localhost/?x=1", "http://localhost/#x",
                "http://localhost/api/chat", "http://localhost:0", "http://localhost:65536", "http://localhost:bad",
                "http://localhost\n", "http://[::ffff:127.0.0.1]", "http://[::1%25lo]", "http://[::1"]
        with patch.object(vlm.urllib.request, "build_opener") as transport:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(vlm.LocalVLMConfigurationError):
                    vlm.analyze_frames(FRAMES, model=MODEL, base_url=url)
            transport.assert_not_called()

    def test_ipv6_loopback_normalization(self):
        self.assertEqual(vlm._local_origin("http://[::1]:11434/"), "http://[::1]:11434")

    def test_proxy_and_redirect_handlers(self):
        def open_request(request, timeout):
            if request.full_url.endswith("/api/tags"):
                return Response({"models": [{"name": MODEL}]})
            if request.full_url.endswith('/api/show'):
                return Response({'capabilities': ['vision']})
            return Response({"done": True, "message": {"content": '{"events": []}'}})
        with patch.object(vlm.urllib.request, "build_opener", return_value=SimpleNamespace(open=open_request)) as build:
            vlm.analyze_frames(FRAMES, model=MODEL)
            handlers = build.call_args.args
            self.assertEqual(handlers[0].proxies, {})
            self.assertIsInstance(handlers[1], vlm._NoRedirect)
            with self.assertRaisesRegex(vlm.LocalVLMBackendError, "redirect"):
                handlers[1].redirect_request(None, None, 307, "", {}, "https://example.invalid")

    def test_model_cloud_alias_rejected_before_images(self):
        with self.assertRaises(vlm.LocalVLMConfigurationError):
            self.call(tags={"models": [{"name": MODEL, "remote_host": "https://ollama.com"}]})
        self.assertEqual(len(self.requests), 1)
        self.assertIsNone(self.requests[0][0].data)

    def test_show_rejects_remote_alias_and_nonvision_before_images(self):
        for show in ({'capabilities': ['vision'], 'remote_model': 'remote-alias'},
                     {'capabilities': ['vision'], 'remote_host': 'https://example.invalid'},
                     {'capabilities': ['vision'], 'details': {'remote_host': 'https://example.invalid'}},
                     {'capabilities': ['completion']}, {}):
            with self.subTest(show=show), self.assertRaises(vlm.LocalVLMConfigurationError):
                self.call(show=show)
            self.assertEqual([request.full_url.rsplit('/', 1)[-1] for request, _ in self.requests], ['tags', 'show'])
            self.assertNotIn('images', json.loads(self.requests[-1][0].data))

    def test_cloud_tag_rejected_without_transport(self):
        with patch.object(vlm.urllib.request, "build_opener") as transport:
            with self.assertRaises(vlm.LocalVLMConfigurationError):
                vlm.analyze_frames(FRAMES, model="qwen3-vl:235b-cloud")
            transport.assert_not_called()

    def test_missing_weights_are_explicit_failure(self):
        with self.assertRaisesRegex(vlm.LocalVLMBackendError, "not installed"):
            self.call(tags={"models": []})
        self.assertEqual(len(self.requests), 1)

    def test_invalid_frame_values(self):
        invalid = [[], FRAMES * 13, [{"timestamp_sec": True, "image_base64": PNG}],
                   [{"timestamp_sec": float("nan"), "image_base64": PNG}],
                   [{"timestamp_sec": -1, "image_base64": PNG}],
                   [{"timestamp_sec": 0, "image_base64": "not base64"}],
                   [{"timestamp_sec": 0, "image_base64": "YWJj"}], FRAMES[::-1], [FRAMES[0], FRAMES[0]]]
        for frames in invalid:
            with self.subTest(frames=str(frames)[:80]), self.assertRaises(vlm.LocalVLMConfigurationError):
                vlm.analyze_frames(frames, model=MODEL)

    def test_request_size_cap_precedes_network(self):
        with patch.object(vlm, "MAX_REQUEST_BYTES", 10), self.assertRaises(vlm.LocalVLMConfigurationError):
            vlm.analyze_frames(FRAMES, model=MODEL)

    def test_invalid_time_and_context_settings(self):
        for value in [0, -1, True, float("inf"), 301]:
            with self.subTest(timeout=value), self.assertRaises(vlm.LocalVLMConfigurationError):
                vlm.analyze_frames(FRAMES, model=MODEL, timeout_sec=value)
        for value in [True, 0, 65536, 4096.0]:
            with self.subTest(num_ctx=value), self.assertRaises(vlm.LocalVLMConfigurationError):
                vlm.analyze_frames(FRAMES, model=MODEL, num_ctx=value)

    def test_unseen_timestamp_and_frame_index_are_rejected(self):
        for changes in [{"timestamp_sec": 6.0}, {"timestamp_sec": 7.0}, {"frame_index": 2},
                        {"frame_index": -1}, {"frame_index": True}]:
            with self.subTest(changes=changes), self.assertRaises(vlm.LocalVLMOutputError):
                self.call({"events": [{**EVENT, **changes}]})

    def test_bad_event_confidence_type_and_properties(self):
        for changes in [{"event": "theft"}, {"event": []}, {"confidence": 1.01}, {"confidence": -0.01},
                        {"confidence": True}, {"description": " "}, {"name": "forbidden field"}]:
            with self.subTest(changes=changes), self.assertRaises(vlm.LocalVLMOutputError):
                self.call({"events": [{**EVENT, **changes}]})

    def test_malformed_and_duplicate_model_json(self):
        for raw in ["not json", '{"events": [], "events": []}', '{"events": [], "score": NaN}', "[]"]:
            with self.subTest(raw=raw), self.assertRaises(vlm.LocalVLMOutputError):
                self.call(outer={"done": True, "message": {"content": raw}})

    def test_incomplete_response_is_not_a_clean_result(self):
        for outer in [{"done": False, "message": {"content": '{"events": []}'}}, {"done": True},
                      {"done": True, "message": {"content": {"events": []}}}]:
            with self.subTest(outer=outer), self.assertRaises(vlm.LocalVLMOutputError):
                self.call(outer=outer)

    def test_generation_limit_rejected_even_if_json_is_valid(self):
        with self.assertRaisesRegex(vlm.LocalVLMOutputError, 'generation limit'):
            self.call(outer={'done': True, 'done_reason': 'length', 'message': {'content': '{"events": []}'}})

    def test_backend_errors_remain_errors(self):
        with self.assertRaises(vlm.LocalVLMBackendError):
            self.call(outer={"error": "private internal details"})
        for error in [urllib.error.URLError("refused"), urllib.error.HTTPError("local", 404, "missing", {}, None)]:
            with patch.object(vlm.urllib.request, "build_opener", return_value=SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(error))):
                with self.assertRaises(vlm.LocalVLMBackendError):
                    vlm.analyze_frames(FRAMES, model=MODEL)

    def test_response_size_limit(self):
        response = Response(b" " * 100, {"Content-Length": str(vlm.MAX_RESPONSE_BYTES + 1)})
        with patch.object(vlm.urllib.request, "build_opener", return_value=SimpleNamespace(open=lambda *a, **k: response)):
            with self.assertRaises(vlm.LocalVLMOutputError):
                vlm.analyze_frames(FRAMES, model=MODEL)
        response = Response(b" " * 101, {})
        with patch.object(vlm, "MAX_RESPONSE_BYTES", 100):
            with patch.object(vlm.urllib.request, "build_opener", return_value=SimpleNamespace(open=lambda *a, **k: response)):
                with self.assertRaises(vlm.LocalVLMOutputError):
                    vlm.analyze_frames(FRAMES, model=MODEL)

    def test_wall_clock_limit_even_when_transport_stalls(self):
        def stalled_open(*args, **kwargs):
            time.sleep(0.25)
            return Response({"models": [{"name": MODEL}]})
        with patch.object(vlm.urllib.request, "build_opener", return_value=SimpleNamespace(open=stalled_open)):
            started = time.monotonic()
            with self.assertRaises(vlm.LocalVLMBackendError):
                vlm.analyze_frames(FRAMES, model=MODEL, timeout_sec=0.03)
            self.assertLess(time.monotonic() - started, 0.2)

    def test_timed_out_worker_blocks_next_call_until_thread_finishes(self):
        entered, release = threading.Event(), threading.Event()
        origin = 'http://127.0.0.1:11434'
        def stalled_open(*args, **kwargs):
            entered.set()
            release.wait(1)
            return Response({'models': [{'name': MODEL}]})
        try:
            with patch.object(vlm.urllib.request, 'build_opener', return_value=SimpleNamespace(open=stalled_open)):
                with self.assertRaises(vlm.LocalVLMBackendError):
                    vlm.analyze_frames(FRAMES, model=MODEL, timeout_sec=.03)
            self.assertTrue(entered.is_set())
            with patch.object(vlm.urllib.request, 'build_opener') as transport:
                with self.assertRaisesRegex(vlm.LocalVLMBackendError, 'still running'):
                    vlm.analyze_frames(FRAMES, model=MODEL)
                transport.assert_not_called()
        finally:
            release.set()
            worker = vlm._ACTIVE_REQUESTS.get(origin)
            if worker is not None:
                worker.join(timeout=2)
        self.assertEqual(self.call({'events': []})['events'], [])

    def test_segment_contact_sheets_keep_original_frame_evidence(self):
        from PIL import Image
        frames = []
        for index in range(8):
            image = Image.new('RGB', (80, 120), (index * 20, 30, 80))
            encoded = io.BytesIO()
            image.save(encoded, format='PNG')
            frames.append({'timestamp_sec': float(index), 'image_base64': base64.b64encode(encoded.getvalue()).decode('ascii')})
        candidate = {**CANDIDATE, 'frame_index': 7, 'timecode_sec': 7.0, 'evidence_frame_indices': [0, 7]}
        result = self.call({'observations': [], 'summary': '', 'candidate_violations': [candidate]},
                           segment=True, frames=frames, image_layout='contact_sheet', frames_per_sheet=4, num_ctx=8192)
        payload = json.loads(self.requests[-1][0].data)
        supplied = json.loads(payload['messages'][1]['content'])
        self.assertEqual(len(payload['messages'][1]['images']), 2)
        self.assertEqual([row['frame_index'] for row in supplied['frame_map']], list(range(8)))
        self.assertEqual(supplied['frame_map'][7]['timestamp_sec'], 7.0)
        self.assertEqual(supplied['frame_map'][7]['sheet_index'], 1)
        self.assertEqual(supplied['frame_map'][7]['row'], 1)
        self.assertEqual(supplied['frame_map'][7]['column'], 1)
        self.assertEqual(result['candidate_violations'][0]['evidence_frame_indices'], [0, 7])
        self.assertEqual(result['inference_meta']['input_frame_count'], 8)
        self.assertEqual(result['inference_meta']['input_image_count'], 2)
        for encoded in payload['messages'][1]['images']:
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as sheet:
                self.assertEqual(sheet.format, 'JPEG')
                self.assertLessEqual(max(sheet.size), 1250)

    def test_impossible_context_rejected_without_dropping_evidence(self):
        frames = [{'timestamp_sec': float(i), 'image_base64': PNG} for i in range(8)]
        with patch.object(vlm.urllib.request, 'build_opener') as transport:
            with self.assertRaisesRegex(vlm.LocalVLMConfigurationError, 'exceed num_ctx'):
                vlm.analyze_segment(frames, model=MODEL, rules=RULES, num_ctx=4096)
            transport.assert_not_called()

    def test_invalid_image_does_not_fall_back_to_individual_transport(self):
        frames = [{'timestamp_sec': 0., 'image_base64': base64.b64encode(b'\xff\xd8\xffinvalid-image').decode('ascii')}]
        with patch.object(vlm.urllib.request, 'build_opener') as transport:
            with self.assertRaises(vlm.LocalVLMConfigurationError):
                vlm.analyze_segment(frames, model=MODEL, rules=RULES, image_layout='contact_sheet', num_ctx=8192)
            transport.assert_not_called()

    def test_segment_candidates_and_catalogue_id_alias(self):
        content = {"observations": [EVENT], "summary": "Сейф виден открытым.", "candidate_violations": [CANDIDATE]}
        result = self.call(content, segment=True, context="Ранее кассир был у стола.")
        self.assertEqual(result["candidate_violations"][0]["timecode_sec"], 7.0)
        payload = json.loads(self.requests[-1][0].data)
        supplied = json.loads(payload["messages"][1]["content"])
        self.assertEqual(supplied["visual_rules"][0]["rule_id"], "safe_left_open")
        self.assertIn("visual_requirements", supplied["visual_rules"][0])
        self.assertNotIn("source", supplied["visual_rules"][0])

    def test_segment_bad_rules_times_and_evidence_rejected(self):
        mutations = [{"rule_id": "made_up"}, {"frame_index": 99}, {"timecode_sec": 6},
                     {"confidence": True}, {"confidence": 1.2}, {"evidence_frame_indices": []},
                     {"evidence_frame_indices": [0]}, {"evidence_frame_indices": [0, 2]},
                     {"evidence_frame_indices": [1, 1]}, {"evidence_frame_indices": [True]},
                     {"description": ""}]
        for changes in mutations:
            content = {"observations": [], "summary": "", "candidate_violations": [{**CANDIDATE, **changes}]}
            with self.subTest(changes=changes), self.assertRaises(vlm.LocalVLMOutputError):
                self.call(content, segment=True)

    def test_segment_oversized_evidence_fails_without_truncation(self):
        frames = [{'timestamp_sec': float(index), 'image_base64': PNG} for index in range(6)]
        candidate = {**CANDIDATE, 'frame_index': 2, 'timecode_sec': 2.0,
                     'evidence_frame_indices': [0, 1, 2, 3]}
        content = {'observations': [], 'summary': '', 'candidate_violations': [candidate]}
        schema = vlm._segment_schema(frames, ['safe_left_open'])
        self.assertEqual(schema['properties']['candidate_violations']['items']['properties']['evidence_frame_indices']['maxItems'], 3)
        with self.assertRaisesRegex(vlm.LocalVLMOutputError, '1 to 3'):
            self.call(content, segment=True, frames=frames, num_ctx=8192)

    def test_three_causal_evidence_indices_and_exact_time_are_preserved(self):
        frames = [{'timestamp_sec': float(index), 'image_base64': PNG} for index in range(6)]
        candidate = {**CANDIDATE, 'frame_index': 3, 'timecode_sec': 3.0,
                     'evidence_frame_indices': [0, 3, 5]}
        result = self.call({'observations': [], 'summary': '', 'candidate_violations': [candidate]},
                           segment=True, frames=frames, num_ctx=8192)
        self.assertEqual(result['candidate_violations'][0]['evidence_frame_indices'], [0, 3, 5])
        self.assertEqual(result['candidate_violations'][0]['timecode_sec'], frames[3]['timestamp_sec'])

    def test_compact_segment_limits_are_strict_in_schema_and_validator(self):
        schema = vlm._segment_schema(FRAMES, ['safe_left_open'])
        self.assertEqual(schema['properties']['observations']['maxItems'], 6)
        self.assertEqual(schema['properties']['observations']['items']['properties']['description']['maxLength'], 160)
        self.assertEqual(schema['properties']['summary']['maxLength'], 360)
        self.assertEqual(schema['properties']['candidate_violations']['maxItems'], 3)
        self.assertEqual(schema['properties']['candidate_violations']['items']['properties']['description']['maxLength'], 240)
        valid = {'observations': [EVENT], 'summary': '', 'candidate_violations': [CANDIDATE]}
        oversized = [
            {**valid, 'observations': [EVENT] * 7},
            {**valid, 'observations': [{**EVENT, 'description': 'x' * 161}]},
            {**valid, 'summary': 'x' * 361},
            {**valid, 'candidate_violations': [CANDIDATE] * 4},
            {**valid, 'candidate_violations': [{**CANDIDATE, 'description': 'x' * 241}]},
        ]
        for content in oversized:
            with self.subTest(field=next(key for key in content if content[key] != valid[key])):
                with self.assertRaises(vlm.LocalVLMOutputError):
                    self.call(content, segment=True)
        result = self.call(valid, segment=True)
        self.assertEqual(result['candidate_violations'][0]['timecode_sec'], 7.0)
        payload = json.loads(self.requests[-1][0].data)
        self.assertEqual(payload['options']['num_predict'], 1536)

    def test_disabled_rules_cannot_generate_candidates(self):
        rules = [{**RULES[0], "enabled": False}]
        content = {"observations": [], "summary": "", "candidate_violations": [CANDIDATE]}
        with self.assertRaises(vlm.LocalVLMOutputError):
            self.call(content, segment=True, rules=rules)


class LoopbackTransportTests(unittest.TestCase):
    def test_real_loopback_http_ignores_environment_proxy_and_blocks_redirect(self):
        requests = []
        redirect = [False]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append(self.path)
                if redirect[0]:
                    self.send_response(307)
                    self.send_header("Location", "http://example.invalid/api/tags")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.reply({"models": [{"name": MODEL}]})

            def do_POST(self):
                requests.append(self.path)
                size = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(size))
                if self.path == '/api/show':
                    self.reply({'capabilities': ['vision']})
                    return
                self.server.received = payload
                self.reply({"done": True, "message": {"content": '{"events": []}'}})

            def reply(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://localhost:{server.server_address[1]}"
                with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""}):
                    try:
                        result = vlm.analyze_frames(FRAMES, model=MODEL, base_url=url, timeout_sec=3)
                    except vlm.LocalVLMBackendError as exc:
                        cause = exc
                        while cause is not None:
                            if isinstance(cause, OSError) and (getattr(cause, "winerror", None) == 10013 or cause.errno == errno.EACCES):
                                self.skipTest("Execution environment blocks loopback TCP connections (EACCES/WinError10013).")
                            # URLError wraps OSError in reason instead of __cause__.
                            if isinstance(cause, urllib.error.URLError) and isinstance(cause.reason, OSError):
                                cause = cause.reason
                            else:
                                cause = cause.__cause__
                        raise
                    self.assertEqual(result["events"], [])
                    self.assertEqual(requests, ["/api/tags", "/api/show", "/api/chat"])
                    self.assertEqual(server.received["messages"][1]["images"], [PNG, PNG])
                    redirect[0] = True
                    with self.assertRaisesRegex(vlm.LocalVLMBackendError, "redirect"):
                        vlm.analyze_frames(FRAMES, model=MODEL, base_url=url, timeout_sec=3)
                    self.assertEqual(requests[-1], "/api/tags")
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
