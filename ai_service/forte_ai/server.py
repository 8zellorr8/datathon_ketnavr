"""Small local HTTP backend for a single video-analysis worker.

This serves an API only. Authentication belongs on every analysis endpoint;
the health endpoint intentionally discloses no paths, keys, or model settings.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import math
import re
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from email import policy
from email.message import Message
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .pipeline import AnalysisError, AnalysisErrorCode
from .local_vlm import LocalVLMOutOfMemoryError, LocalVLMOutputError, LocalVLMTimeoutError


MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_PENDING_JOBS = 2  # One executing job plus one waiting job for the single GPU.
MAX_COMPLETED_JOBS = 100
COMPLETED_TTL_SECONDS = 3600
JOB_TIMEOUT_SECONDS = 300
_REJECT_DRAIN_SECONDS = 2.0
_JOB_PATH = re.compile(r"/api/v1/jobs/([0-9a-f]{32})")
_BOUNDARY = re.compile(r"[0-9A-Za-z'()+_,./:=? -]{1,70}")
_INDEX_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"
_FRONTEND_ASSETS = {
    '/styles.css': 'text/css; charset=utf-8',
    '/app.js': 'text/javascript; charset=utf-8',
    '/analysis-client.js': 'text/javascript; charset=utf-8',
    '/demo.js': 'text/javascript; charset=utf-8',
    '/pdf-export.js': 'text/javascript; charset=utf-8',
    '/vendor/pdf-lib.min.js': 'text/javascript; charset=utf-8',
    '/vendor/fontkit.umd.min.js': 'text/javascript; charset=utf-8',
    '/vendor/THIRD-PARTY.txt': 'text/plain; charset=utf-8',
    '/vendor/pdf-lib.LICENSE.md': 'text/plain; charset=utf-8',
    '/vendor/fontkit.LICENSE.md': 'text/plain; charset=utf-8',
    '/assets/cash-desk.svg': 'image/svg+xml',
    '/assets/fonts/NotoSans-Regular.ttf': 'font/ttf',
    '/assets/fonts/NotoSans-Bold.ttf': 'font/ttf',
    '/assets/fonts/NotoSans.LICENSE.txt': 'text/plain; charset=utf-8',
}
_MAX_STATIC_BYTES = 8 * 1024 * 1024
_SAFE_ANALYSIS_FAILURES = {
    AnalysisErrorCode.ANALYSIS_FAILED: "Video analysis failed. See local diagnostics.",
    AnalysisErrorCode.TIME_BUDGET_EXCEEDED: "Analysis exceeded its configured processing budget. No verdict was produced.",
    AnalysisErrorCode.CANDIDATE_BUDGET_EXCEEDED: "Too many candidate events for the configured review budget. No verdict was produced.",
    AnalysisErrorCode.EVIDENCE_BUDGET_EXCEEDED: "The required evidence exceeds the supported frame budget. No verdict was produced.",
    AnalysisErrorCode.INVALID_ANALYSIS_CONFIGURATION: "The local video analysis configuration is invalid.",
    AnalysisErrorCode.INVALID_RULE_CATALOGUE: "The local rule catalogue is invalid.",
    AnalysisErrorCode.NO_VERIFIED_RULES: "No enabled rules with verified normative sources are configured.",
    AnalysisErrorCode.VIDEO_DURATION_EXCEEDED: "The video exceeds the configured maximum duration.",
}


class _RequestError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


@dataclass
class _Job:
    id: str
    path: Path
    deadline: float
    status: str = "processing"
    result: dict = field(default_factory=dict)
    future: Future | None = None
    terminal_at: float | None = None


def _delete_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # A timed-out analyzer may still hold the file on Windows. Its finally
        # block retries after the analyzer releases it.
        pass


def _default_analyzer(video_path, config, *, deadline):
    from .pipeline import analyze_video

    return analyze_video(video_path, config, deadline=deadline)


def _validate_analysis_result(result: dict) -> None:
    """Reject malformed model/pipeline results before they reach API clients."""
    if not isinstance(result, dict) or result.get("status") != "done":
        raise ValueError("Analyzer did not return a completed result")
    verdict = result.get("verdict")
    violations = result.get("violations")
    if verdict not in ("violation", "no_violation") or not isinstance(violations, list):
        raise ValueError("Analyzer result has an invalid verdict or violations list")
    if (verdict == "violation") != bool(violations):
        raise ValueError("Analyzer verdict contradicts its violations list")

    def valid_time(value):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0
        )

    for event in violations:
        if not isinstance(event, dict):
            raise ValueError("Each violation must be an object")
        if not isinstance(event.get("type"), str) or not event["type"].strip():
            raise ValueError("Each violation needs a nonempty type")
        if not valid_time(event.get("timecode_sec")):
            raise ValueError("Each violation needs a finite nonnegative timecode")
        if not isinstance(event.get("description"), str):
            raise ValueError("Each violation needs a description string")
        if "verification_status" in event:
            verification_status = event["verification_status"]
            if verification_status not in ("normative_source_verified", "needs_regulation_review"):
                raise ValueError("Each event verification status must be supported")
            if event.get("requires_human_review") is not True:
                raise ValueError("Verified-source and observation events both require human review")
            if verification_status == "needs_regulation_review":
                if "rule_source" not in event or event["rule_source"] is not None:
                    raise ValueError("Unverified observations must not include a normative citation")
                if event.get("detection_basis") != "visual_observation":
                    raise ValueError("Unverified observations need a visual-observation detection basis")
            else:
                source = event.get("rule_source")
                if not isinstance(source, dict) or source.get("verified") is not True:
                    raise ValueError("Verified-source events need a verified normative source")
                if event.get("detection_basis") != "verified_rule":
                    raise ValueError("Verified-source events need a verified-rule detection basis")
                # This status verifies the source only; human review is still
                # required before drawing a legal or disciplinary conclusion.
        if "regulatory_note" in event and not isinstance(event["regulatory_note"], str):
            raise ValueError("Event regulatory notes must be strings")
        frames = event.get("evidence_frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise ValueError("Each violation needs one to three evidence frames")
        for frame in frames:
            if not isinstance(frame, dict) or not valid_time(frame.get("timestamp_sec")):
                raise ValueError("Each evidence frame needs a finite nonnegative timestamp")
            encoded = frame.get("image_base64")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("Each evidence frame needs a base64 JPEG string")
            try:
                jpeg = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Invalid base64 evidence image") from exc
            if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                raise ValueError("Evidence images must be JPEG data")


class _JobManager:
    def __init__(self, config: dict, analyzer: Callable | None):
        self.config = dict(config)
        self.analyzer = analyzer or _default_analyzer
        self.runtime_dir = Path(config.get("runtime_dir", "data/runtime")).expanduser().resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.config["runtime_dir"] = str(self.runtime_dir)
        self.lock = threading.RLock()
        self.jobs: dict[str, _Job] = {}
        self.reservations = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forte-analysis")
        self.stopped = threading.Event()
        self.monitor = threading.Thread(target=self._monitor, name="forte-job-deadlines", daemon=True)
        self.monitor.start()

    def _prune(self, now: float) -> None:
        finished = [
            job for job in self.jobs.values()
            if job.terminal_at is not None and (job.future is None or job.future.done())
        ]
        finished.sort(key=lambda job: job.terminal_at)
        expired = {job.id for job in finished if now - job.terminal_at >= COMPLETED_TTL_SECONDS}
        remaining = [job for job in finished if job.id not in expired]
        overflow = max(0, len(remaining) - MAX_COMPLETED_JOBS)
        expired.update(job.id for job in remaining[:overflow])
        for job_id in expired:
            _delete_upload(self.jobs[job_id].path)
            del self.jobs[job_id]

    def reserve(self) -> bool:
        with self.lock:
            self._prune(time.monotonic())
            inflight = sum(job.future is not None and not job.future.done() for job in self.jobs.values())
            if self.stopped.is_set() or inflight + self.reservations >= MAX_PENDING_JOBS:
                return False
            self.reservations += 1
            return True

    def release(self) -> None:
        with self.lock:
            self.reservations -= 1

    @staticmethod
    def _failure(job: _Job, code: str, message: str) -> None:
        job.status = "failed"
        job.result = {"error": {"code": code, "message": message}}
        job.terminal_at = time.monotonic()

    def submit_reserved(self, path: Path, deadline: float, job_id: str) -> None:
        with self.lock:
            job = _Job(job_id, path, deadline)
            self.jobs[job_id] = job
            try:
                job.future = self.executor.submit(self._run, job)
            except RuntimeError:
                del self.jobs[job_id]
                raise
            finally:
                self.reservations -= 1

    def _expire(self, job: _Job, now: float) -> None:
        if job.status == "processing" and now >= job.deadline:
            self._failure(job, "analysis_timeout", "Analysis exceeded the five-minute deadline.")
            if job.future is not None:
                job.future.cancel()
            _delete_upload(job.path)

    def _run(self, job: _Job) -> None:
        try:
            with self.lock:
                self._expire(job, time.monotonic())
                if job.status != "processing":
                    return
            result = self.analyzer(job.path, self.config, deadline=job.deadline)
            _validate_analysis_result(result)
            # Validate serializability before an HTTP reader sees the result.
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            with self.lock:
                self._expire(job, time.monotonic())
                if job.status == "processing":
                    job.status = "done"
                    job.result = dict(result)
                    job.terminal_at = time.monotonic()
        except Exception as exc:
            # Exception messages may contain local paths or sensitive model
            # input. Trust only a known enum on an internal AnalysisError;
            # arbitrary exception text/attributes never reach logs or clients.
            safe_code = AnalysisErrorCode.ANALYSIS_FAILED.value
            safe_message = _SAFE_ANALYSIS_FAILURES[AnalysisErrorCode.ANALYSIS_FAILED]
            if isinstance(exc, (LocalVLMOutOfMemoryError, MemoryError)):
                safe_code = "model_out_of_memory"
                safe_message = "The local model could not allocate enough RAM or GPU memory. No verdict was produced."
            elif isinstance(exc, LocalVLMTimeoutError):
                safe_code = "model_timeout"
                safe_message = "The local model request exceeded its time limit. No verdict was produced."
            elif isinstance(exc, LocalVLMOutputError):
                safe_code = "model_invalid_output"
                safe_message = "The local model returned an invalid or inconsistent result. No verdict was produced."
            elif isinstance(exc, AnalysisError):
                candidate_code = getattr(exc, "safe_code", None)
                if type(candidate_code) is AnalysisErrorCode and candidate_code in _SAFE_ANALYSIS_FAILURES:
                    safe_code = candidate_code.value
                    safe_message = _SAFE_ANALYSIS_FAILURES[candidate_code]
            print(f"Analysis job {job.id} failed: {type(exc).__name__} safe_code={safe_code}",
                  file=sys.stderr, flush=True)
            with self.lock:
                if job.status == "processing":
                    self._failure(job, safe_code, safe_message)
        finally:
            _delete_upload(job.path)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            now = time.monotonic()
            self._prune(now)
            job = self.jobs.get(job_id)
            if job is None:
                return None
            self._expire(job, now)
            return {**job.result, "job_id": job.id, "status": job.status}

    def _monitor(self) -> None:
        while not self.stopped.wait(0.2):
            with self.lock:
                now = time.monotonic()
                for job in list(self.jobs.values()):
                    self._expire(job, now)
                self._prune(now)

    def close(self) -> None:
        self.stopped.set()
        self.monitor.join(timeout=1)
        with self.lock:
            for job in self.jobs.values():
                if job.future is not None:
                    job.future.cancel()
                if job.status == "processing":
                    self._failure(job, "server_stopped", "The analysis server stopped.")
                _delete_upload(job.path)
        # A running native model call cannot safely be killed by Python. Its
        # cooperative deadline remains in force; queued futures are cancelled.
        self.executor.shutdown(wait=False, cancel_futures=True)


def _extract_mov(body: bytes, content_type: str) -> bytes:
    if "\r" in content_type or "\n" in content_type:
        raise _RequestError(400, "invalid_multipart", "Invalid multipart Content-Type.")
    header = Message()
    header["Content-Type"] = content_type
    boundary = header.get_param("boundary", header="content-type")
    if header.get_content_type() != "multipart/form-data":
        raise _RequestError(415, "unsupported_media_type", "Use multipart/form-data with field 'file'.")
    if not isinstance(boundary, str) or not _BOUNDARY.fullmatch(boundary) or boundary.endswith(" "):
        raise _RequestError(400, "invalid_multipart", "Missing or invalid multipart boundary.")
    try:
        prefix = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
    except UnicodeEncodeError as exc:
        raise _RequestError(400, "invalid_multipart", "Invalid multipart Content-Type.") from exc
    message = BytesParser(policy=policy.default).parsebytes(prefix + body)
    if not message.is_multipart() or any(part.defects for part in message.walk()):
        raise _RequestError(400, "invalid_multipart", "Malformed multipart request.")
    parts = list(message.iter_parts())
    if len(parts) > 16 or any(part.is_multipart() for part in parts):
        raise _RequestError(400, "invalid_multipart", "Unsupported multipart structure.")
    files = [
        part for part in parts
        if part.get_content_disposition() == "form-data"
        and part.get_param("name", header="content-disposition") == "file"
    ]
    if len(files) != 1:
        raise _RequestError(400, "invalid_file_field", "Provide exactly one file field named 'file'.")
    filename = files[0].get_filename()
    if not filename or not filename.casefold().endswith(".mov"):
        raise _RequestError(400, "invalid_extension", "The uploaded file must have a .mov extension.")
    payload = files[0].get_payload(decode=True)
    if not isinstance(payload, bytes) or not payload:
        raise _RequestError(400, "empty_file", "The uploaded video is empty.")
    return payload


class AnalysisHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, manager: _JobManager):
        self.job_manager = manager
        super().__init__(address, handler)

    def server_close(self):
        try:
            super().server_close()
        finally:
            self.job_manager.close()


def _handler_factory(manager: _JobManager, api_key: str):
    expected_key = api_key.encode("utf-8")
    frontend_setting = manager.config.get('frontend_dir')
    frontend_root = Path(frontend_setting).resolve() if frontend_setting else None
    if frontend_root is not None and not (frontend_root / 'index.html').is_file():
        raise ValueError('Configured frontend is unavailable.')

    class Handler(BaseHTTPRequestHandler):
        server_version = "ForteAI"
        sys_version = ""

        def log_message(self, _format, *_args):
            # No request headers, query strings, filenames, or credentials.
            pass

        def _json(self, status: int, payload: dict, *, close: bool = False) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def _reject_unread_body(self, status: int, code: str, message: str, length: int) -> None:
            """Deliver an early error before closing a bounded upload socket.

            Closing with incoming bytes still in flight can reset TCP before a
            browser reads the error. Send the complete response, half-close,
            then discard through peer EOF with fixed byte and time limits.
            No discarded bytes are parsed, retained, saved, or analyzed.
            """
            self.close_connection = True
            try:
                self._json(status, {"error": {"code": code, "message": message}}, close=True)
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                deadline = time.monotonic() + _REJECT_DRAIN_SECONDS
                remaining = length
                while remaining >= 0:
                    seconds_left = deadline - time.monotonic()
                    if seconds_left <= 0:
                        break
                    self.connection.settimeout(seconds_left)
                    # Once the declared body ends, wait for peer EOF. Closing
                    # immediately at Content-Length still races response ACKs.
                    # One extra byte bounds clients that exceed their length.
                    chunk = self.rfile.read1(min(64 * 1024, remaining + 1))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except (OSError, TimeoutError):
                # The error was already sent. Disconnects/timeouts must not
                # produce a second response or disclose request contents.
                pass

        def _static(self, location: Path, content_type: str, *, root: Path) -> None:
            try:
                location = location.resolve()
                if not location.is_relative_to(root.resolve()) or location.stat().st_size > _MAX_STATIC_BYTES:
                    raise OSError('Unsupported frontend asset')
                content = location.read_bytes()
            except OSError:
                self._error(503, "web_unavailable", "The local web interface is unavailable.")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "font-src 'self'; media-src 'self' blob:; worker-src 'self' blob:; "
                "connect-src 'self'; base-uri 'none'; object-src 'none'; "
                "frame-ancestors 'none'; form-action 'self'",
            )
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _index(self) -> None:
            if frontend_root is None:
                self._static(_INDEX_PATH, 'text/html; charset=utf-8', root=_INDEX_PATH.parent)
            else:
                self._static(frontend_root / 'index.html', 'text/html; charset=utf-8', root=frontend_root)

        def _authorized(self) -> bool:
            keys = self.headers.get_all("X-API-Key", [])
            provided = keys[0].encode("utf-8") if len(keys) == 1 else b""
            if not hmac.compare_digest(provided, expected_key):
                # Authentication still precedes multipart parsing and queue
                # admission. Drain only a single, valid, bounded length; never
                # accept malformed framing or an oversized unauthorized body.
                lengths = self.headers.get_all("Content-Length", [])
                if (self.command == "POST" and self.headers.get("Transfer-Encoding") is None
                        and len(lengths) == 1 and re.fullmatch(r"[0-9]+", lengths[0])):
                    try:
                        length = int(lengths[0])
                    except ValueError:
                        length = 0
                    if 0 < length <= MAX_UPLOAD_BYTES:
                        self._reject_unread_body(401, "unauthorized", "A valid X-API-Key is required.", length)
                        return False
                self._error(401, "unauthorized", "A valid X-API-Key is required.")
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._index()
                return
            if path == "/health":
                self._json(200, {"status": "ok", "backend": "local", "workers": 1,
                                 **({'ui': 'fortelens'} if frontend_root is not None else {})})
                return
            if path.startswith("/api/v1/jobs/"):
                if not self._authorized():
                    return
                match = _JOB_PATH.fullmatch(path)
                result = manager.get(match.group(1)) if match else None
                if result is None:
                    self._error(404, "job_not_found", "Job not found or expired.")
                else:
                    self._json(200, result)
                return
            if frontend_root is not None and path in _FRONTEND_ASSETS:
                self._static(frontend_root / path.lstrip('/'), _FRONTEND_ASSETS[path], root=frontend_root)
                return
            self._error(404, "not_found", "Endpoint not found.")

        def do_POST(self):
            received_at = time.monotonic()
            if urlsplit(self.path).path != "/api/v1/analyze":
                self._error(404, "not_found", "Endpoint not found.")
                return
            if not self._authorized():
                return
            if self.headers.get("Transfer-Encoding") is not None:
                self._error(400, "unsupported_transfer_encoding", "Use Content-Length without Transfer-Encoding.")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                self._error(411, "length_required", "Content-Length is required.")
                return
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                self._error(400, "invalid_length", "Invalid Content-Length.")
                return
            length = int(lengths[0])
            if length > MAX_UPLOAD_BYTES:
                self._error(413, "upload_too_large", "Multipart upload exceeds 64 MiB.")
                return
            if length <= 0:
                self._error(400, "empty_upload", "The request body is empty.")
                return
            if not manager.reserve():
                self._reject_unread_body(429, "queue_full", "The analysis queue is full. Try again later.", length)
                return
            reserved = True
            upload_path = None
            deadline = received_at + JOB_TIMEOUT_SECONDS
            try:
                chunks = []
                remaining = length
                while remaining:
                    seconds_left = deadline - time.monotonic()
                    if seconds_left <= 0:
                        raise _RequestError(408, "upload_timeout", "Upload exceeded the analysis deadline.")
                    self.connection.settimeout(min(30, seconds_left))
                    # read1 returns after one underlying read. Checking the
                    # absolute deadline each time also bounds slow-drip uploads.
                    chunk = self.rfile.read1(min(1024 * 1024, remaining))
                    if not chunk:
                        raise _RequestError(400, "incomplete_upload", "The request body is incomplete.")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                body = b"".join(chunks)
                payload = _extract_mov(body, self.headers.get("Content-Type", ""))
                if time.monotonic() >= deadline:
                    raise _RequestError(408, "upload_timeout", "Upload exceeded the analysis deadline.")
                job_id = uuid.uuid4().hex
                upload_path = manager.runtime_dir / f"{job_id}.mov"
                with upload_path.open("xb") as stream:
                    stream.write(payload)
                # submit_reserved always consumes the reservation, even if the
                # executor has been closed while an upload was being received.
                reserved = False
                manager.submit_reserved(upload_path, deadline, job_id)
                self._json(202, {"job_id": job_id, "status": "processing"})
            except _RequestError as exc:
                self._error(exc.status, exc.code, exc.message)
                if upload_path is not None:
                    _delete_upload(upload_path)
            except TimeoutError:
                self._error(408, "upload_timeout", "The upload timed out.")
                if upload_path is not None:
                    _delete_upload(upload_path)
            except (OSError, RuntimeError):
                self._error(503, "server_unavailable", "The local analysis server is unavailable.")
                if upload_path is not None:
                    _delete_upload(upload_path)
            finally:
                if reserved:
                    manager.release()

    return Handler


def build_server(
    config: dict, *, host: str = "127.0.0.1", port: int = 8000,
    api_key: str, analyzer=None,
) -> AnalysisHTTPServer:
    """Build an unstarted HTTP server; use port=0 for an ephemeral test port."""
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("A nonempty API key is required")
    manager = _JobManager(config, analyzer)
    try:
        return AnalysisHTTPServer((host, port), _handler_factory(manager, api_key), manager)
    except Exception:
        manager.close()
        raise


def serve(
    config: dict, *, host: str = "127.0.0.1", port: int = 8000,
    api_key: str, analyzer=None,
) -> None:
    """Run the local backend until interrupted. API keys are never printed."""
    server = build_server(config, host=host, port=port, api_key=api_key, analyzer=analyzer)
    print(f"ForteBank local backend listening on http://{host}:{server.server_port}")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
