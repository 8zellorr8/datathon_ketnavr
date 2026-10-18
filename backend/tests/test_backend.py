"""Exercise the backend with a deterministic analyzer, without model weights.

The HTTP tests use the actual multipart parser, worker, authentication and
frontend files. The tiny JPEG fixture tests the transport contract only; model
quality and image decoding remain the AI service's responsibility.
"""

from __future__ import annotations

import base64
import copy
import http.client
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn

from ai_service.forte_ai.pipeline import AnalysisError, AnalysisErrorCode
from ai_service.forte_ai.server import MAX_UPLOAD_BYTES
from backend.__main__ import check_configuration
from backend.app import create_application
from backend.errors import CatalogueError, InvalidAnalysisResultError
from backend.ports import VideoAnalyzer
from backend.repository import DictionaryRegulationRepository
from backend.service import AnalysisService


KEY = "backend-integration-test-key"
JPEG = base64.b64encode(b"\xff\xd8transport-fixture\xff\xd9").decode("ascii")
ACTIVE_IDS = {
    "unattended_cash", "cash_not_secured_between_clients",
    "count_after_client_exit", "cash_issued_without_count",
    "uncounted_unsealed_cash", "personal_items_on_desk",
}


def positive_result(rule_id="unattended_cash"):
    return {
        "status": "done", "verdict": "violation",
        "analysis": {"duration_sec": 12.0, "elapsed_sec": 0.1},
        "violations": [{
            "rule_id": rule_id,
            "type": "Уход кассира до завершения обслуживания",
            "timecode_sec": 2.5,
            "description": "Клиент остаётся у кассы; кассир покидает рабочее место.",
            "evidence_frames": [{"timestamp_sec": 2.5, "image_base64": JPEG}],
            "verification_status": "normative_source_verified",
            "requires_human_review": True,
            "detection_basis": "verified_rule",
            "rule_source": {
                "title": "Untrusted model citation", "section": "invented 999",
                "url": "https://untrusted.example/999", "verified": True,
            },
        }],
    }


def negative_result():
    return {"status": "done", "verdict": "no_violation", "violations": []}


class StubAnalyzer(VideoAnalyzer):
    """Return the original object so tests can detect accidental mutation."""

    def __init__(self, result=None, *, error=None):
        self.result = negative_result() if result is None else result
        self.error = error
        self.calls = []

    def analyze(self, video_path, config, *, deadline):
        self.calls.append((Path(video_path), copy.deepcopy(config), deadline))
        if self.error is not None:
            raise self.error
        return self.result


class RegulationRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.repository = DictionaryRegulationRepository()

    def test_catalogue_covers_enabled_ai_rules_and_unknown_has_no_match(self):
        self.assertIsInstance(self.repository.all(), tuple)
        for rule_id in ACTIVE_IDS:
            with self.subTest(rule_id=rule_id):
                self.assertIsNotNone(self.repository.find(rule_id))
                self.assertIsInstance(self.repository.catalogue(rule_id), dict)
        self.assertIsNone(self.repository.find("not_a_confirmed_rule"))
        self.assertIsNone(self.repository.catalogue("not_a_confirmed_rule"))

    def test_serialized_catalogue_cannot_modify_the_repository(self):
        expected = self.repository.catalogue(None)
        detached = self.repository.catalogue(None)
        detached.clear()
        self.assertEqual(self.repository.catalogue(None), expected)
        record = self.repository.catalogue("unattended_cash")
        expected_record = copy.deepcopy(record)
        record.clear()
        self.assertEqual(self.repository.catalogue("unattended_cash"), expected_record)


class ConfigurationCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rules_path = Path(self.temp.name) / "rules.json"
        source = DictionaryRegulationRepository().find("unattended_cash").source()
        self.valid_entry = {"id": "unattended_cash", "enabled": True, "source": source}

    def check(self, entries):
        self.rules_path.write_text(json.dumps(entries), encoding="utf-8")
        return check_configuration({"rules_path": str(self.rules_path)})

    def test_rejects_non_boolean_enabled_or_source_verification(self):
        for field, value in (("enabled", "false"), ("enabled", 1),
                             ("verified", "true"), ("verified", 1)):
            with self.subTest(field=field, value=value):
                entry = copy.deepcopy(self.valid_entry)
                if field == "enabled":
                    entry[field] = value
                else:
                    entry["source"][field] = value
                with self.assertRaises(CatalogueError):
                    self.check([entry])

    def test_rejects_non_object_rule_or_missing_source_verification(self):
        malformed_entries = ["not-an-object"]
        for source in (None, [], "verified", {}, {"verified": None}):
            entry = copy.deepcopy(self.valid_entry)
            entry["source"] = source
            malformed_entries.append(entry)
        for entry in malformed_entries:
            with self.subTest(entry=entry):
                with self.assertRaises(CatalogueError):
                    self.check([entry])
        with self.assertRaises(CatalogueError):
            self.check({"rules": [self.valid_entry]})

    def test_rejects_changed_paragraph_for_known_rule(self):
        entry = copy.deepcopy(self.valid_entry)
        entry["source"]["section"] = "Пункт 999 — неподтверждённое основание"
        with self.assertRaises(CatalogueError):
            self.check([entry])

    def test_rejects_catalogue_without_enabled_verified_rules(self):
        disabled = copy.deepcopy(self.valid_entry)
        disabled["enabled"] = False
        unverified = copy.deepcopy(self.valid_entry)
        unverified["source"]["verified"] = False
        for entries in ([], [disabled], [unverified]):
            with self.subTest(entries=entries):
                with self.assertRaises(CatalogueError):
                    self.check(entries)

    def test_unknown_active_ids_are_flagged_for_review_without_inference(self):
        unknown = copy.deepcopy(self.valid_entry)
        unknown["id"] = "new_team_rule"
        with patch("backend.analyzer.LocalAIAnalyzer.analyze", side_effect=AssertionError("Preflight ran inference")):
            result = self.check([self.valid_entry, unknown])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ai_rules_requiring_review"], ["new_team_rule"])
        self.assertIs(result["model_inference_performed"], False)


class AnalysisServiceTests(unittest.TestCase):
    def setUp(self):
        self.repository = DictionaryRegulationRepository()

    def analyze(self, raw):
        analyzer = StubAnalyzer(raw)
        service = AnalysisService(analyzer, self.repository)
        return service.analyze(Path("fixture.mov"), {}, deadline=time.monotonic() + 30)

    def test_trusted_dictionary_replaces_model_citation_without_changing_evidence(self):
        raw = positive_result()
        snapshot = copy.deepcopy(raw)
        result = self.analyze(raw)
        event = result["violations"][0]
        self.assertEqual(raw, snapshot, "Enrichment must not mutate the analyzer's result")
        self.assertEqual(event["evidence_frames"], snapshot["violations"][0]["evidence_frames"])
        self.assertEqual(event["timecode_sec"], 2.5)
        self.assertEqual(event["description"], snapshot["violations"][0]["description"])
        self.assertEqual(event["rule_source"]["url"], "https://adilet.zan.kz/rus/docs/V1900019680")
        self.assertIs(event["rule_source"]["verified"], True)
        self.assertNotIn("invented", json.dumps(event["rule_source"]))
        self.assertNotIn("untrusted.example", json.dumps(result))
        self.assertEqual(event["verification_status"], "normative_source_verified")
        self.assertIs(event["requires_human_review"], True)

    def test_unknown_disabled_or_missing_id_never_acquires_a_legal_citation(self):
        for rule_id in ("invented_rule", "safe_left_open", "key_left_in_lock", None):
            with self.subTest(rule_id=rule_id):
                raw = positive_result(rule_id)
                if rule_id is None:
                    del raw["violations"][0]["rule_id"]
                result = self.analyze(raw)
                event = result["violations"][0]
                self.assertEqual(result["verdict"], "violation")
                self.assertIsNone(event["rule_source"])
                self.assertEqual(event["verification_status"], "needs_regulation_review")
                self.assertEqual(event["detection_basis"], "visual_observation")
                self.assertIs(event["requires_human_review"], True)
                self.assertTrue(event["regulatory_note"])
                self.assertEqual(event["evidence_frames"], raw["violations"][0]["evidence_frames"])

    def test_known_id_does_not_promote_an_explicitly_unverified_observation(self):
        raw = positive_result()
        raw["violations"][0].update(
            verification_status="needs_regulation_review", detection_basis="visual_observation",
            rule_source=None,
        )
        result = self.analyze(raw)
        event = result["violations"][0]
        self.assertIsNone(event["rule_source"])
        self.assertEqual(event["verification_status"], "needs_regulation_review")
        self.assertIs(event["requires_human_review"], True)

    def test_multiple_events_keep_order_and_separate_citations(self):
        raw = positive_result()
        raw["violations"].append(positive_result("personal_items_on_desk")["violations"][0])
        raw["violations"].append(positive_result("unknown")["violations"][0])
        result = self.analyze(raw)
        self.assertEqual([event["rule_id"] for event in result["violations"]],
                         ["unattended_cash", "personal_items_on_desk", "unknown"])
        self.assertNotEqual(result["violations"][0]["rule_source"]["section"],
                            result["violations"][1]["rule_source"]["section"])
        self.assertIsNone(result["violations"][2]["rule_source"])

    def test_callable_passes_video_config_and_deadline_to_analyzer(self):
        analyzer = StubAnalyzer()
        service = AnalysisService(analyzer, self.repository)
        path = Path("fixture.mov")
        config = {"example": "configuration"}
        deadline = time.monotonic() + 30
        result = service(path, config, deadline=deadline)
        self.assertEqual(result["verdict"], "no_violation")
        self.assertEqual(result["violations"], [])
        self.assertEqual(analyzer.calls, [(path, config, deadline)])

    def test_invalid_result_raises_typed_error(self):
        examples = [
            {"status": "done", "verdict": "no_violation", "violations": [{}]},
            positive_result(), positive_result(), positive_result(),
        ]
        examples[1]["violations"][0]["rule_id"] = ["unattended_cash"]
        examples[2]["analysis"]["duration_sec"] = float("nan")
        examples[3]["violations"][0]["verification_status"] = "definitely_illegal"
        for raw in examples:
            with self.subTest(result=raw):
                with self.assertRaises(InvalidAnalysisResultError):
                    self.analyze(raw)

    def test_expired_deadline_prevents_running_the_analyzer(self):
        analyzer = StubAnalyzer()
        service = AnalysisService(analyzer, self.repository)
        with self.assertRaises(AnalysisError) as captured:
            service(Path("fixture.mov"), {}, deadline=time.monotonic() - 1)
        self.assertEqual(captured.exception.safe_code, AnalysisErrorCode.TIME_BUDGET_EXCEEDED)
        self.assertEqual(analyzer.calls, [])

    def test_late_negative_result_cannot_replace_an_expired_deadline(self):
        analyzer = StubAnalyzer()
        service = AnalysisService(analyzer, self.repository)
        with patch("backend.service.time.monotonic", side_effect=(10.0, 21.0)):
            with self.assertRaises(AnalysisError) as captured:
                service(Path("fixture.mov"), {}, deadline=20.0)
        self.assertEqual(captured.exception.safe_code, AnalysisErrorCode.TIME_BUDGET_EXCEEDED)
        self.assertEqual(len(analyzer.calls), 1)


def multipart(payload=b"fake-mov-for-backend-contract"):
    boundary = "forte-backend-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="clip.mov"\r\n'
        "Content-Type: video/quicktime\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class BackendHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.servers = []
        self.addCleanup(self.stop_servers)

    def stop_servers(self):
        for fixture in self.servers:
            fixture.server.should_exit = True
            fixture.thread.join(timeout=3)
            if fixture.thread.is_alive():
                fixture.server.force_exit = True
                fixture.thread.join(timeout=2)
            fixture.listener.close()
            self.assertFalse(fixture.thread.is_alive(), "Uvicorn failed to shut down")

    def start_server(self, analyzer):
        runtime = Path(self.temp.name) / str(len(self.servers))
        application = create_application(
            {"runtime_dir": str(runtime)}, api_key=KEY, port=0, analyzer=analyzer,
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            application, host="127.0.0.1", port=port, loop="asyncio",
            lifespan="on", log_level="error", access_log=False,
        ))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True,
        )
        fixture = SimpleNamespace(
            application=application, server=server, server_port=port,
            thread=thread, listener=listener,
        )
        self.servers.append(fixture)
        thread.start()
        until = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < until:
            time.sleep(0.01)
        self.assertTrue(server.started, "Uvicorn failed to start with application lifespan")
        return fixture

    def request(self, server, method, path, *, body=None, headers=None, raw=False):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=dict(headers or {}))
            response = connection.getresponse()
            payload = response.read()
            response_headers = {name.lower(): value for name, value in response.getheaders()}
            return response.status, (payload if raw else json.loads(payload)), response_headers
        finally:
            connection.close()

    def upload(self, server, *, ui_headers=None):
        body, content_type = multipart()
        headers = {"Content-Type": content_type}
        headers.update({"X-API-Key": KEY} if ui_headers is None else ui_headers)
        path = "/api/v1/analyze" if ui_headers is None else "/ui/api/v1/analyze"
        return self.request(server, "POST", path, body=body, headers=headers)

    def wait_job(self, server, job_id, *, ui_headers=None):
        path = f"/api/v1/jobs/{job_id}"
        headers = {"X-API-Key": KEY}
        if ui_headers is not None:
            path = "/ui" + path
            headers = ui_headers
        until = time.monotonic() + 3
        while time.monotonic() < until:
            status, result, _ = self.request(server, "GET", path, headers=headers)
            self.assertEqual(status, 200)
            if result["status"] in ("done", "failed"):
                return result
            time.sleep(0.01)
        self.fail("Stub analysis did not finish within three seconds")

    def session(self, server):
        status, body, headers = self.request(server, "GET", "/", raw=True)
        self.assertEqual(status, 200)
        self.assertNotIn(KEY.encode(), body)
        self.assertIn("httponly", headers["set-cookie"].lower())
        self.assertIn("samesite=strict", headers["set-cookie"].lower())
        return {
            "Cookie": headers["set-cookie"].split(";", 1)[0],
            "Origin": f"http://127.0.0.1:{server.server_port}",
            "Sec-Fetch-Site": "same-origin",
        }

    def test_upload_to_poll_enriches_positive_result_from_dictionary(self):
        analyzer = StubAnalyzer(positive_result())
        server = self.start_server(analyzer)
        status, accepted, _ = self.upload(server)
        self.assertEqual(status, 202)
        self.assertEqual(accepted["status"], "processing")
        result = self.wait_job(server, accepted["job_id"])
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["verdict"], "violation")
        event = result["violations"][0]
        self.assertEqual(event["rule_source"]["url"], "https://adilet.zan.kz/rus/docs/V1900019680")
        self.assertEqual(event["evidence_frames"][0]["image_base64"], JPEG)
        self.assertNotIn("untrusted.example", json.dumps(result))
        self.assertGreater(analyzer.calls[0][2], time.monotonic())

    def test_negative_result_remains_empty_and_has_no_synthetic_violation(self):
        server = self.start_server(StubAnalyzer())
        status, accepted, _ = self.upload(server)
        self.assertEqual(status, 202)
        result = self.wait_job(server, accepted["job_id"])
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["verdict"], "no_violation")
        self.assertEqual(result["violations"], [])

    def test_third_upload_is_rejected_while_worker_and_queue_are_busy(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        class BlockingAnalyzer(StubAnalyzer):
            def analyze(self, video_path, config, *, deadline):
                started.set()
                if not release.wait(5):
                    raise RuntimeError("Test worker was not released")
                return super().analyze(video_path, config, deadline=deadline)

        server = self.start_server(BlockingAnalyzer())
        status, first, _ = self.upload(server)
        self.assertEqual(status, 202)
        self.assertTrue(started.wait(2), "First analysis did not start")
        status, second, _ = self.upload(server)
        self.assertEqual(status, 202)
        manager = server.application.state.jobs
        files_before = set(manager.runtime_dir.iterdir())
        status, rejected, _ = self.upload(server)
        self.assertEqual(status, 429)
        self.assertEqual(rejected["error"]["code"], "queue_full")
        self.assertEqual(len(manager.jobs), 2)
        self.assertEqual(manager.reservations, 0)
        self.assertEqual(set(manager.runtime_dir.iterdir()), files_before)
        release.set()
        for accepted in (first, second):
            self.assertEqual(self.wait_job(server, accepted["job_id"])["status"], "done")

    def test_invalid_and_oversized_uploads_release_admission_without_files(self):
        analyzer = StubAnalyzer()
        server = self.start_server(analyzer)
        valid_body, content_type = multipart()
        empty_body, _ = multipart(payload=b"")
        cases = [
            (None, {"Content-Length": str(MAX_UPLOAD_BYTES + 1), "Content-Type": content_type},
             413, "upload_too_large"),
            (b"broken multipart", {"Content-Type": "multipart/form-data; boundary=missing"},
             400, "invalid_multipart"),
            (b"{}", {"Content-Type": "application/json"}, 415, "unsupported_media_type"),
            (valid_body.replace(b"clip.mov", b"clip.exe"), {"Content-Type": content_type},
             400, "invalid_extension"),
            (empty_body, {"Content-Type": content_type}, 400, "empty_file"),
        ]
        manager = server.application.state.jobs
        for body, headers, expected_status, code in cases:
            with self.subTest(code=code):
                status, result, _ = self.request(
                    server, "POST", "/api/v1/analyze", body=body,
                    headers={"X-API-Key": KEY, **headers},
                )
                self.assertEqual(status, expected_status)
                self.assertEqual(result["error"]["code"], code)
                self.assertEqual(manager.reservations, 0)
                self.assertEqual(manager.jobs, {})
                self.assertEqual(list(manager.runtime_dir.iterdir()), [])
        self.assertEqual(analyzer.calls, [])
        status, accepted, _ = self.upload(server)
        self.assertEqual(status, 202, "Rejected requests must not consume queue capacity")
        self.assertEqual(self.wait_job(server, accepted["job_id"])["status"], "done")

    def test_exception_is_sanitized_and_never_reported_as_no_violation(self):
        server = self.start_server(StubAnalyzer(error=RuntimeError("PRIVATE_TOKEN_AND_PATH")))
        with patch("sys.stderr", io.StringIO()):
            status, accepted, _ = self.upload(server)
            self.assertEqual(status, 202)
            result = self.wait_job(server, accepted["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "analysis_failed")
        self.assertNotIn("verdict", result)
        self.assertNotIn("violations", result)
        self.assertNotIn("PRIVATE_TOKEN_AND_PATH", json.dumps(result))
        self.assertNotIn(KEY, json.dumps(result))

    def test_malformed_or_out_of_bounds_evidence_fails_instead_of_clean_verdict(self):
        examples = [
            {"status": "done", "verdict": "violation", "violations": []},
            positive_result(), positive_result(), positive_result(),
        ]
        examples[1]["violations"][0]["timecode_sec"] = 13
        examples[2]["violations"][0]["evidence_frames"][0]["timestamp_sec"] = 13
        examples[3]["violations"][0]["evidence_frames"][0]["image_base64"] = "not base64"
        for raw in examples:
            with self.subTest(result=raw):
                server = self.start_server(StubAnalyzer(raw))
                with patch("sys.stderr", io.StringIO()):
                    status, accepted, _ = self.upload(server)
                    self.assertEqual(status, 202)
                    result = self.wait_job(server, accepted["job_id"])
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("verdict", result)
                self.assertNotIn("violations", result)

    def test_catalogue_requires_auth_and_returns_dictionary_records(self):
        server = self.start_server(StubAnalyzer())
        repository = DictionaryRegulationRepository()
        for suffix in ("", "/unattended_cash"):
            path = "/api/v1/regulations" + suffix
            for headers in ({}, {"X-API-Key": "wrong"}):
                status, _, _ = self.request(server, "GET", path, headers=headers)
                self.assertEqual(status, 401)
            status, result, _ = self.request(server, "GET", path, headers={"X-API-Key": KEY})
            self.assertEqual(status, 200)
            self.assertEqual(result, repository.catalogue(suffix[1:] or None))
        status, result, _ = self.request(
            server, "GET", "/api/v1/regulations/unknown", headers={"X-API-Key": KEY},
        )
        self.assertEqual(status, 404)
        self.assertEqual(result["error"]["code"], "regulation_not_found")

    def test_catalogue_is_read_only_and_cannot_expose_project_paths(self):
        server = self.start_server(StubAnalyzer())
        status, before, _ = self.request(
            server, "GET", "/api/v1/regulations", headers={"X-API-Key": KEY},
        )
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            server, "POST", "/api/v1/regulations", body=b"{}",
            headers={"X-API-Key": KEY, "Content-Type": "application/json"},
        )
        self.assertIn(status, (404, 405))
        for path in ("/backend/regulations.py", "/config/local.json", "/api/v1/regulations/../../README.md"):
            status, result, _ = self.request(server, "GET", path, headers={"X-API-Key": KEY})
            self.assertEqual(status, 404)
            self.assertNotIn(self.temp.name, json.dumps(result))
        _, after, _ = self.request(server, "GET", "/api/v1/regulations", headers={"X-API-Key": KEY})
        self.assertEqual(after, before)

    def test_browser_session_upload_catalogue_and_job_ownership(self):
        server = self.start_server(StubAnalyzer(positive_result()))
        owner = self.session(server)
        stranger = self.session(server)
        self.assertNotEqual(owner["Cookie"], stranger["Cookie"])
        status, _, _ = self.request(server, "GET", "/ui/api/v1/regulations", headers=owner)
        self.assertEqual(status, 200)
        status, accepted, _ = self.upload(server, ui_headers=owner)
        self.assertEqual(status, 202)
        result = self.wait_job(server, accepted["job_id"], ui_headers=owner)
        self.assertEqual(result["status"], "done")
        status, _, _ = self.request(
            server, "GET", f"/ui/api/v1/jobs/{accepted['job_id']}", headers=stranger,
        )
        self.assertEqual(status, 404)
        status, _, _ = self.request(server, "GET", "/ui/api/v1/regulations")
        self.assertEqual(status, 403)

    def test_browser_cross_origin_or_missing_origin_cannot_submit_video(self):
        analyzer = StubAnalyzer()
        server = self.start_server(analyzer)
        session = self.session(server)
        for changed in (
            {**session, "Origin": "https://attacker.example"},
            {key: value for key, value in session.items() if key != "Origin"},
            {**session, "Sec-Fetch-Site": "cross-site"},
        ):
            status, _, _ = self.upload(server, ui_headers=changed)
            self.assertEqual(status, 403)
        self.assertEqual(analyzer.calls, [])

    def test_health_and_frontend_are_served_without_server_secrets(self):
        server = self.start_server(StubAnalyzer())
        status, health, _ = self.request(server, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["backend_service"], "forte_backend")
        self.assertEqual(health["framework"], "fastapi")
        self.assertEqual(health["ui"], "fortelens")
        self.assertNotIn(KEY, json.dumps(health))
        self.assertNotIn(self.temp.name, json.dumps(health))
        for path in ("/", "/app.js", "/analysis-client.js", "/pdf-export.js", "/styles.css"):
            with self.subTest(path=path):
                status, body, headers = self.request(server, "GET", path, raw=True)
                self.assertEqual(status, 200)
                self.assertTrue(body)
                self.assertNotIn(KEY.encode(), body)
                self.assertEqual(headers["x-content-type-options"], "nosniff")

    def test_swagger_and_openapi_describe_authenticated_multipart_api(self):
        server = self.start_server(StubAnalyzer())
        status, document, headers = self.request(server, "GET", "/docs", raw=True)
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertIn(b"/openapi.json", document)
        self.assertNotIn(KEY.encode(), document)
        status, schema, _ = self.request(server, "GET", "/openapi.json")
        self.assertEqual(status, 200)
        schemes = schema["components"]["securitySchemes"]
        key_schemes = {
            name for name, value in schemes.items()
            if value.get("type") == "apiKey" and value.get("in") == "header"
            and value.get("name") == "X-API-Key"
        }
        self.assertTrue(key_schemes, "OpenAPI must document the X-API-Key security scheme")
        for path, method in (("/api/v1/analyze", "post"), ("/api/v1/jobs/{job_id}", "get"),
                             ("/api/v1/regulations", "get"), ("/api/v1/regulations/{rule_id}", "get")):
            operation = schema["paths"][path][method]
            self.assertTrue(any(key_schemes.intersection(requirement) for requirement in operation["security"]))
        upload = schema["paths"]["/api/v1/analyze"]["post"]
        multipart_schema = upload["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertIn("file", multipart_schema["required"])
        self.assertEqual(multipart_schema["properties"]["file"]["format"], "binary")
        self.assertFalse(any(path.startswith("/ui/") for path in schema["paths"]))


if __name__ == "__main__":
    unittest.main()
