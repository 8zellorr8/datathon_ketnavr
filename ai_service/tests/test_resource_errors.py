"""Bounded local resource failures, using mocked transports and analyzers only."""

import io
import json
import socket
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from forte_ai import local_vlm as vlm
from forte_ai import server as api


PRIVATE = "PRIVATE_FRAME_DESCRIPTION_AND_LOCAL_FILENAME"


class _Body(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


class _Response(io.BytesIO):
    status = 200

    def __init__(self, content):
        raw = json.dumps(content).encode("utf-8")
        super().__init__(raw)
        self.headers = {"Content-Length": str(len(raw))}


class ResourceErrorTests(unittest.TestCase):
    def request(self, opener):
        return vlm._request_json(
            opener, "http://127.0.0.1:11435/api/chat", b"{}", time.monotonic() + 2,
        )

    def test_http_oom_body_is_bounded_and_not_echoed(self):
        body = _Body(("CUDA allocation failed: out of memory " + PRIVATE).encode() + b"x" * 20000)
        error = urllib.error.HTTPError("local", 500, PRIVATE, {}, body)
        opener = SimpleNamespace(open=Mock(side_effect=error))
        with self.assertRaises(vlm.LocalVLMOutOfMemoryError) as caught:
            self.request(opener)
        self.assertEqual(body.read_sizes, [8192])
        self.assertTrue(body.closed)
        self.assertIsInstance(caught.exception, vlm.LocalVLMBackendError)
        self.assertEqual(str(caught.exception), "Local Ollama ran out of memory.")
        self.assertNotIn(PRIVATE, str(caught.exception))

    def test_json_backend_oom_variants_are_private(self):
        for error in (
            "model requires more system memory than is available " + PRIVATE,
            "insufficient GPU memory " + PRIVATE,
            "cudaMalloc failed " + PRIVATE,
            {"message": "std::bad_alloc " + PRIVATE},
        ):
            with self.subTest(error_type=type(error).__name__):
                opener = SimpleNamespace(open=Mock(return_value=_Response({"error": error})))
                with self.assertRaises(vlm.LocalVLMOutOfMemoryError) as caught:
                    self.request(opener)
                self.assertNotIn(PRIVATE, str(caught.exception))

    def test_socket_and_wrapped_socket_timeouts_are_typed(self):
        for error in (socket.timeout(PRIVATE), urllib.error.URLError(socket.timeout(PRIVATE))):
            with self.subTest(error_type=type(error).__name__):
                opener = SimpleNamespace(open=Mock(side_effect=error))
                with self.assertRaises(vlm.LocalVLMTimeoutError) as caught:
                    self.request(opener)
                self.assertIsInstance(caught.exception, vlm.LocalVLMBackendError)
                self.assertNotIn(PRIVATE, str(caught.exception))

    def test_unrelated_http_failure_is_not_out_of_memory(self):
        body = _Body(("CUDA illegal memory access " + PRIVATE).encode())
        error = urllib.error.HTTPError("local", 500, PRIVATE, {}, body)
        with self.assertRaises(vlm.LocalVLMBackendError) as caught:
            self.request(SimpleNamespace(open=Mock(side_effect=error)))
        self.assertIs(type(caught.exception), vlm.LocalVLMBackendError)
        self.assertNotIn(PRIVATE, str(caught.exception))
        self.assertEqual(body.read_sizes, [8192])

    def test_expired_deadline_is_typed(self):
        with self.assertRaises(vlm.LocalVLMTimeoutError):
            vlm._remaining(time.monotonic() - 1)

    def test_contact_sheet_timeout_is_not_configuration_error(self):
        with patch("forte_ai.image_packing.pack_contact_sheets", side_effect=TimeoutError(PRIVATE)):
            with self.assertRaises(vlm.LocalVLMTimeoutError) as caught:
                vlm._segment_images([], "contact_sheet", 6, 1250, time.monotonic() + 2)
        self.assertNotIn(PRIVATE, str(caught.exception))

    def check_failed_job(self, error, expected_code):
        def broken(*_args, **_kwargs):
            raise error

        with tempfile.TemporaryDirectory() as temporary:
            manager = api._JobManager({"runtime_dir": temporary}, broken)
            try:
                upload = Path(temporary) / "upload.mov"
                upload.write_bytes(b"mock video, never decoded")
                job = api._Job("a" * 32, upload, time.monotonic() + 5)
                manager.jobs[job.id] = job
                captured = io.StringIO()
                with patch.object(api.sys, "stderr", captured):
                    manager._run(job)
                result = manager.get(job.id)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], expected_code)
                self.assertNotIn("verdict", result)
                self.assertNotIn("violations", result)
                self.assertNotIn(PRIVATE, json.dumps(result))
                self.assertNotIn(PRIVATE, captured.getvalue())
                self.assertIn("safe_code=" + expected_code, captured.getvalue())
                self.assertFalse(upload.exists())
            finally:
                manager.close()

    def test_server_model_timeout_is_failed_without_verdict(self):
        self.check_failed_job(vlm.LocalVLMTimeoutError(PRIVATE), "model_timeout")

    def test_server_invalid_model_output_is_failed_without_verdict(self):
        self.check_failed_job(vlm.LocalVLMOutputError(PRIVATE), "model_invalid_output")

    def test_server_model_and_native_memory_errors_have_safe_code(self):
        for error in (vlm.LocalVLMOutOfMemoryError(PRIVATE), MemoryError(PRIVATE)):
            with self.subTest(error_type=type(error).__name__):
                self.check_failed_job(error, "model_out_of_memory")


if __name__ == "__main__":
    unittest.main()
