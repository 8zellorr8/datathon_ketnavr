"""Exercise the real loopback HTTP API with small stub analyzers."""

import http.client
import base64
import copy
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from forte_ai import server as api
from forte_ai.pipeline import AnalysisError, AnalysisErrorCode


KEY = "test-local-api-key"


def multipart(filename="clip.mov", payload=b"fake-mov-for-api-test"):
    boundary = "forte-test-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: video/quicktime\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name) / "uploads"
        self.servers = []
        self.addCleanup(self.stop_servers)

    def stop_servers(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def start_server(self, analyzer):
        server = api.build_server(
            {"runtime_dir": str(self.runtime)}, port=0, api_key=KEY, analyzer=analyzer,
        )
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return server

    def request(self, server, method, path, body=None, headers=None, *, add_length=True, decode_json=True):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.putrequest(method, path)
            headers = dict(headers or {})
            if body is not None and add_length and "Content-Length" not in headers:
                headers["Content-Length"] = str(len(body))
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders(body)
            response = connection.getresponse()
            payload = response.read()
            if decode_json:
                return response.status, json.loads(payload)
            return response.status, {
                "headers": {name.lower(): value for name, value in response.getheaders()},
                "body": payload,
            }
        except PermissionError as exc:
            self.skipTest(f"Loopback connections are blocked by this environment: {exc}")
        finally:
            connection.close()

    def upload(self, server, *, filename="clip.mov"):
        body, content_type = multipart(filename)
        return self.request(server, "POST", "/api/v1/analyze", body, {
            "X-API-Key": KEY, "Content-Type": content_type,
        })

    def wait_for_job(self, server, job_id, terminal=True):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            status, result = self.request(server, "GET", f"/api/v1/jobs/{job_id}", headers={"X-API-Key": KEY})
            self.assertEqual(status, 200)
            if not terminal or result["status"] in {"done", "failed"}:
                return result
            time.sleep(0.01)
        self.fail("Stub analysis did not finish")

    def test_upload_async_result_and_safe_filename_cleanup(self):
        started = threading.Event()
        release = threading.Event()
        seen = {}

        def analyzer(path, config, *, deadline):
            seen.update(path=Path(path), deadline=deadline)
            self.assertEqual(Path(path).read_bytes(), b"fake-mov-for-api-test")
            self.assertGreater(deadline, time.monotonic())
            started.set()
            release.wait(2)
            return {"status": "done", "verdict": "no_violation", "violations": []}

        self.addCleanup(release.set)
        server = self.start_server(analyzer)
        status, response = self.upload(server, filename="../../private-name.mov")
        self.assertEqual(status, 202)
        self.assertTrue(started.wait(1))
        job_id = response["job_id"]
        self.assertEqual(seen["path"].name, f"{job_id}.mov")
        self.assertEqual(seen["path"].parent, self.runtime.resolve())
        self.assertEqual(self.wait_for_job(server, job_id, terminal=False)["status"], "processing")
        release.set()
        result = self.wait_for_job(server, job_id)
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["verdict"], "no_violation")
        until = time.monotonic() + 1
        while seen["path"].exists() and time.monotonic() < until:
            time.sleep(0.01)
        self.assertFalse(seen["path"].exists())

    def test_authentication_required_for_upload_and_job_status(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        body, content_type = multipart()
        for key in (None, "wrong-key"):
            headers = {"Content-Type": content_type}
            if key is not None:
                headers["X-API-Key"] = key
            status, response = self.request(server, "POST", "/api/v1/analyze", body, headers)
            self.assertEqual(status, 401)
            self.assertNotIn(KEY, json.dumps(response))
            status, _ = self.request(server, "GET", "/api/v1/jobs/" + "0" * 32, headers=headers)
            self.assertEqual(status, 401)
        status, health = self.request(server, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertNotIn("runtime_dir", health)

    def test_malformed_multipart_and_wrong_extension_are_rejected(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        for content_type in ("multipart/form-data", "multipart/form-data; boundary=missing", "application/json"):
            status, _ = self.request(server, "POST", "/api/v1/analyze", b"not-multipart", {
                "X-API-Key": KEY, "Content-Type": content_type,
            })
            self.assertIn(status, (400, 415))
        status, result = self.upload(server, filename="clip.exe")
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_extension")
        self.assertEqual(list(self.runtime.glob("*.mov")), [])

    def test_analyzer_exception_returns_sanitized_failure(self):
        def broken(*_args, **_kwargs):
            failure = RuntimeError("SENSITIVE_LOCAL_PATH_AND_SECRET")
            # A foreign exception cannot impersonate a known internal failure.
            failure.safe_code = AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED
            raise failure

        captured = io.StringIO()
        with patch.object(api.sys, "stderr", captured):
            server = self.start_server(broken)
            status, response = self.upload(server)
            self.assertEqual(status, 202)
            result = self.wait_for_job(server, response["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "analysis_failed")
        self.assertNotIn("SENSITIVE", json.dumps(result))
        self.assertNotIn("verdict", result)
        self.assertIn("RuntimeError safe_code=analysis_failed", captured.getvalue())
        self.assertNotIn("SENSITIVE", captured.getvalue())
        self.assertNotIn(KEY, captured.getvalue())

    def test_known_analysis_failure_codes_are_private_and_have_no_verdict(self):
        for code in (AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED,
                     AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED,
                     AnalysisErrorCode.TIME_BUDGET_EXCEEDED):
            with self.subTest(code=code):
                def broken(*_args, _code=code, **_kwargs):
                    raise AnalysisError("PRIVATE_MODEL_DESCRIPTION_AND_FILENAME", safe_code=_code)

                captured = io.StringIO()
                with patch.object(api.sys, "stderr", captured):
                    server = self.start_server(broken)
                    status, response = self.upload(server)
                    self.assertEqual(status, 202)
                    result = self.wait_for_job(server, response["job_id"])
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], code.value)
                self.assertEqual(result["error"]["message"], api._SAFE_ANALYSIS_FAILURES[code])
                self.assertNotIn("verdict", result)
                self.assertNotIn("violations", result)
                self.assertNotIn("PRIVATE", json.dumps(result))
                self.assertIn("AnalysisError safe_code=" + code.value, captured.getvalue())
                self.assertNotIn("PRIVATE", captured.getvalue())
                self.assertNotIn(KEY, captured.getvalue())

    def test_modified_analysis_error_code_does_not_leak_text(self):
        def broken(*_args, **_kwargs):
            failure = AnalysisError("PRIVATE_EXCEPTION_MESSAGE")
            failure.safe_code = "PRIVATE_MODEL_TEXT_IN_CODE"
            raise failure

        captured = io.StringIO()
        with patch.object(api.sys, "stderr", captured):
            server = self.start_server(broken)
            _, response = self.upload(server)
            result = self.wait_for_job(server, response["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "analysis_failed")
        self.assertNotIn("verdict", result)
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertIn("safe_code=analysis_failed", captured.getvalue())
        self.assertNotIn("PRIVATE", captured.getvalue())

    def test_valid_violation_result_and_invalid_schema(self):
        valid = {
            "status": "done", "verdict": "violation", "violations": [{
                "type": "Наблюдаемое событие", "timecode_sec": 2.5,
                "description": "Описание видимого действия",
                "evidence_frames": [{
                    "timestamp_sec": 2.5,
                    # Signature-shaped JPEG fixture; the real image is decoded
                    # by FFmpeg in pipeline tests, outside this HTTP contract.
                    "image_base64": base64.b64encode(b"\xff\xd8fixture\xff\xd9").decode(),
                }],
            }],
        }
        server = self.start_server(lambda *_args, **_kwargs: valid)
        status, response = self.upload(server)
        self.assertEqual(status, 202)
        self.assertEqual(self.wait_for_job(server, response["job_id"])["verdict"], "violation")
        malformed = []
        for verdict, violations in (("unknown", []), ("violation", []), ("no_violation", valid["violations"]), ("no_violation", {})):
            malformed.append({"status": "done", "verdict": verdict, "violations": violations})
        for field, value in (("type", None), ("description", 42), ("timecode_sec", -1), ("timecode_sec", float("nan")), ("evidence_frames", []), ("evidence_frames", [valid["violations"][0]["evidence_frames"][0]] * 4)):
            bad = copy.deepcopy(valid)
            bad["violations"][0][field] = value
            malformed.append(bad)
        for field, value in (("timestamp_sec", float("inf")), ("timestamp_sec", True), ("image_base64", "not-base64"), ("image_base64", base64.b64encode(b"not-jpeg").decode())):
            bad = copy.deepcopy(valid)
            bad["violations"][0]["evidence_frames"][0][field] = value
            malformed.append(bad)
        for bad in malformed:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    api._validate_analysis_result(bad)

    def test_malformed_completed_result_becomes_failed_job(self):
        server = self.start_server(lambda *_args, **_kwargs: {
            "status": "done", "verdict": "violation", "violations": [],
        })
        _, response = self.upload(server)
        result = self.wait_for_job(server, response["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "analysis_failed")
        self.assertNotIn("verdict", result)

    def test_queue_limit_and_deadline_fail_without_negative_verdict(self):
        started = threading.Event()
        release = threading.Event()

        def blocked(*_args, **_kwargs):
            started.set()
            release.wait()
            return {"status": "done", "verdict": "no_violation", "violations": []}

        self.addCleanup(release.set)
        server = self.start_server(blocked)
        first_status, first = self.upload(server)
        self.assertTrue(started.wait(3), "The first worker must be running before the queue is filled")
        second_status, second = self.upload(server)
        self.assertEqual((first_status, second_status), (202, 202))
        # Exercise an ordinary upload, like browser fetch, without Expect.
        # Early rejection must remain a readable HTTP error despite its body.
        body, content_type = multipart()
        status, rejected = self.request(server, "POST", "/api/v1/analyze", body, headers={
            "X-API-Key": KEY, "Content-Type": content_type,
        })
        self.assertEqual(status, 429)
        self.assertEqual(rejected["error"]["code"], "queue_full")
        # Expire both jobs explicitly after the queue assertion instead of
        # racing three HTTP exchanges against an arbitrary 150 ms wall clock.
        manager = server.job_manager
        with manager.lock:
            first_job = manager.jobs[first["job_id"]]
            second_job = manager.jobs[second["job_id"]]
            first_job.deadline = second_job.deadline = time.monotonic() - 1
        for job in (first, second):
            result = self.wait_for_job(server, job["job_id"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["code"], "analysis_timeout")
            self.assertNotIn("verdict", result)
        self.assertTrue(second_job.future.cancelled(), "The waiting job must not start after expiry")
        self.assertFalse(first_job.future.done(), "The running worker is still blocked after timeout")
        release.set()
        first_job.future.result(timeout=3)
        late_result = self.wait_for_job(server, first["job_id"])
        self.assertEqual(late_result["status"], "failed")
        self.assertEqual(late_result["error"]["code"], "analysis_timeout")
        self.assertNotIn("verdict", late_result, "A late clean result must never replace the timeout")

    def test_queue_full_body_errors_are_readable_without_parsing_or_new_jobs(self):
        started = threading.Event()
        release = threading.Event()

        def blocked(*_args, **_kwargs):
            started.set()
            release.wait()
            return {"status": "done", "verdict": "no_violation", "violations": []}

        self.addCleanup(release.set)
        server = self.start_server(blocked)
        self.assertEqual(self.upload(server)[0], 202)
        self.assertTrue(started.wait(3))
        self.assertEqual(self.upload(server)[0], 202)
        manager = server.job_manager
        previous_files = set(self.runtime.iterdir())
        with patch.object(api, "_extract_mov", side_effect=AssertionError("Rejected body must not be parsed")):
            for size_mib in (1, 8, 63):
                with self.subTest(size_mib=size_mib):
                    body, content_type = multipart(payload=b"x" * (size_mib * 1024 * 1024))
                    status, response = self.request(server, "POST", "/api/v1/analyze", body, {
                        "X-API-Key": KEY, "Content-Type": content_type,
                    }, decode_json=False)
                    self.assertEqual(status, 429)
                    self.assertEqual(response["headers"]["connection"], "close")
                    payload = json.loads(response["body"])
                    self.assertEqual(payload["error"]["code"], "queue_full")
                    self.assertNotIn("verdict", payload)
                    self.assertNotIn(KEY, response["body"].decode())
                    self.assertEqual(len(manager.jobs), 2)
                    self.assertEqual(manager.reservations, 0)
                    self.assertEqual(set(self.runtime.iterdir()), previous_files)

    def test_unauthorized_body_error_is_readable_without_parsing_or_admission(self):
        server = self.start_server(lambda *_args, **_kwargs: self.fail("Unauthorized analysis ran"))
        body, content_type = multipart(payload=b"x" * (1024 * 1024))
        with patch.object(api, "_extract_mov", side_effect=AssertionError("Unauthorized body must not be parsed")):
            with patch.object(server.job_manager, "reserve", side_effect=AssertionError("Unauthorized request reserved a job")):
                for _ in range(5):
                    status, response = self.request(server, "POST", "/api/v1/analyze", body, {
                        "X-API-Key": "wrong-fixture-key", "Content-Type": content_type,
                    }, decode_json=False)
                    self.assertEqual(status, 401)
                    self.assertEqual(response["headers"]["connection"], "close")
                    payload = json.loads(response["body"])
                    self.assertEqual(payload["error"]["code"], "unauthorized")
                    self.assertNotIn("verdict", payload)
                    self.assertNotIn("wrong-fixture-key", response["body"].decode())
        self.assertEqual(server.job_manager.jobs, {})
        self.assertEqual(server.job_manager.reservations, 0)
        self.assertEqual(list(self.runtime.iterdir()), [])

    def test_early_error_without_body_finishes_within_fixed_drain_deadline(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        # Two uploads reserving slots are sufficient to fill admission capacity.
        self.assertTrue(server.job_manager.reserve())
        self.assertTrue(server.job_manager.reserve())
        self.addCleanup(server.job_manager.release)
        self.addCleanup(server.job_manager.release)
        finished = threading.Event()
        original = server.RequestHandlerClass._reject_unread_body

        def observed(handler, *args):
            try:
                return original(handler, *args)
            finally:
                finished.set()

        # Keep the client write side open after reading the response. The
        # server must finish even if a client never sends its declared body.
        with patch.object(server.RequestHandlerClass, "_reject_unread_body", observed):
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=3) as connection:
                request = (
                    "POST /api/v1/analyze HTTP/1.0\r\nHost: localhost\r\n"
                    f"X-API-Key: {KEY}\r\nContent-Length: 1048576\r\n"
                    "Content-Type: multipart/form-data; boundary=fixture\r\n\r\n"
                ).encode()
                begin = time.monotonic()
                connection.sendall(request)
                received = b""
                while b"\r\n\r\n" not in received:
                    chunk = connection.recv(4096)
                    self.assertTrue(chunk, "Server closed before sending the error headers")
                    received += chunk
                self.assertTrue(received.startswith(b"HTTP/1.0 429"))
                self.assertLess(time.monotonic() - begin, 1.0, "Error must arrive before drain timeout")
                self.assertTrue(finished.wait(api._REJECT_DRAIN_SECONDS + 1), "Discard must have an absolute deadline")
                self.assertLess(time.monotonic() - begin, api._REJECT_DRAIN_SECONDS + 1)
        self.assertEqual(server.job_manager.jobs, {})
        self.assertEqual(list(self.runtime.iterdir()), [])

    def test_unauthorized_malformed_or_oversized_lengths_do_not_enter_drain(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        with patch.object(server.RequestHandlerClass, "_reject_unread_body", side_effect=AssertionError("Invalid framing entered drain")):
            for length in (None, "0", "-1", "invalid", str(api.MAX_UPLOAD_BYTES + 1)):
                headers = {"X-API-Key": "wrong-fixture-key"}
                if length is not None:
                    headers["Content-Length"] = length
                status, result = self.request(server, "POST", "/api/v1/analyze", headers=headers)
                self.assertEqual(status, 401)
                self.assertEqual(result["error"]["code"], "unauthorized")
            status, result = self.request(server, "POST", "/api/v1/analyze", headers={
                "X-API-Key": "wrong-fixture-key", "Content-Length": "1048576", "Transfer-Encoding": "chunked",
            })
            self.assertEqual(status, 401)
            self.assertEqual(result["error"]["code"], "unauthorized")

    def test_missing_length_and_oversize_body_headers(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        for length, expected in ((None, 411), (str(api.MAX_UPLOAD_BYTES + 1), 413)):
            headers = {"X-API-Key": KEY, "Content-Type": "multipart/form-data; boundary=test"}
            if length is not None:
                headers["Content-Length"] = length
            status, _ = self.request(server, "POST", "/api/v1/analyze", headers=headers, add_length=False)
            self.assertEqual(status, expected)

    def test_frontend_pages_are_same_origin_html_without_embedded_key(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        pages = []
        for path in ("/", "/index.html", "/?source=local"):
            status, response = self.request(server, "GET", path, decode_json=False)
            self.assertEqual(status, 200)
            headers, content = response["headers"], response["body"]
            self.assertEqual(headers["content-type"], "text/html; charset=utf-8")
            self.assertEqual(int(headers["content-length"]), len(content))
            self.assertEqual(headers["cache-control"], "no-store")
            self.assertEqual(headers["x-content-type-options"], "nosniff")
            self.assertIn("connect-src 'self'", headers["content-security-policy"])
            document = content.decode("utf-8")
            self.assertIn('<html lang="ru">', document)
            self.assertIn("/api/v1/analyze", document)
            self.assertIn("/api/v1/jobs/", document)
            self.assertIn("Нарушение не обнаружено", document)
            self.assertIn("Обнаружены признаки возможных нарушений", document)
            self.assertNotIn(KEY, document)
            self.assertNotIn("localStorage", document)
            self.assertNotIn("sessionStorage", document)
            pages.append(content)
        self.assertEqual(pages[0], pages[1])
        self.assertEqual(pages[1], pages[2])

    def test_frontend_does_not_expose_arbitrary_project_files(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        for path in ("/static/index.html", "/static/../../config/local.json", "/config/local.json", "/forte_ai/server.py"):
            status, result = self.request(server, "GET", path)
            self.assertEqual(status, 404)
            self.assertEqual(result["error"]["code"], "not_found")

    def test_missing_frontend_returns_sanitized_error(self):
        server = self.start_server(lambda *_args, **_kwargs: {})
        with patch.object(api, "_INDEX_PATH", Path(self.temp.name) / "missing-ui.html"):
            status, result = self.request(server, "GET", "/")
        self.assertEqual(status, 503)
        self.assertEqual(result["error"]["code"], "web_unavailable")
        self.assertNotIn(self.temp.name, json.dumps(result))


class _MemoryConnection:
    """A socket-shaped byte transport for the actual HTTP request handler."""

    def __init__(self, request):
        self.incoming = io.BytesIO(request)
        self.outgoing = bytearray()

    def makefile(self, _mode, _buffering):
        return self.incoming

    def sendall(self, data):
        self.outgoing.extend(data)

    def settimeout(self, _timeout):
        pass

    def shutdown(self, _how):
        pass


class InMemoryServerTests(ServerTests):
    """Run the same contract checks where OS policy disables loopback clients.

    This still runs BaseHTTPRequestHandler's real header parser, routing,
    multipart parser, job worker, deadlines, authentication, and serialization.
    It is deliberately a separate suite; it does not claim socket verification.
    """

    def request(self, server, method, path, body=None, headers=None, *, add_length=True, decode_json=True):
        headers = dict(headers or {})
        if body is not None and add_length and "Content-Length" not in headers:
            headers["Content-Length"] = str(len(body))
        request = f"{method} {path} HTTP/1.0\r\nHost: localhost\r\n".encode()
        request += "".join(f"{key}: {value}\r\n" for key, value in headers.items()).encode("latin-1")
        request += b"\r\n" + (body or b"")
        connection = _MemoryConnection(request)
        server.RequestHandlerClass(connection, ("127.0.0.1", 12345), server)
        header, payload = bytes(connection.outgoing).split(b"\r\n\r\n", 1)
        status = int(header.split(b" ", 2)[1])
        if decode_json:
            return status, json.loads(payload)
        response_headers = {}
        for line in header.split(b"\r\n")[1:]:
            name, value = line.decode("latin-1").split(":", 1)
            response_headers[name.lower()] = value.strip()
        return status, {"headers": response_headers, "body": payload}


if __name__ == "__main__":
    unittest.main()
