"""FastAPI composition root; one local inference worker per application process."""
from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from ai_service.forte_ai.server import (
    FRONTEND_ASSETS, JOB_TIMEOUT_SECONDS, MAX_STATIC_BYTES, MAX_UPLOAD_BYTES,
    JobManager, UploadError, extract_video_upload,
)

from .analyzer import LocalAIAnalyzer
from .auth import SessionAuthenticator
from .http_errors import APIError
from .ports import VideoAnalyzer
from .repository import DictionaryRegulationRepository
from .service import AnalysisService

ROOT = Path(__file__).resolve().parents[1]
_JOB_ID = re.compile(r"[0-9a-f]{32}")
_RULE_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CSP = ("default-src 'none'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; font-src 'self'; media-src 'self' blob:; worker-src 'self' blob:; "
        "connect-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'")
_UPLOAD_SCHEMA = {"requestBody": {"required": True, "content": {"multipart/form-data": {
    "schema": {"type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary", "description": "MOV or MP4; multipart body up to 64 MiB"}
    }}
}}}}


class AnalysisController:
    """HTTP validation and orchestration; domain analysis stays in AnalysisService."""

    def __init__(self, regulations: DictionaryRegulationRepository, auth: SessionAuthenticator):
        self._regulations = regulations
        self._auth = auth
        self.router = APIRouter(dependencies=[Depends(auth.authorize)])
        for prefix in ("/api/v1", "/ui/api/v1"):
            visible = prefix == "/api/v1"
            self.router.add_api_route(prefix + "/analyze", self.upload, methods=["POST"],
                                      status_code=202, openapi_extra=_UPLOAD_SCHEMA, include_in_schema=visible)
            self.router.add_api_route(prefix + "/jobs/{job_id}", self.job, methods=["GET"],
                                      include_in_schema=visible)
            self.router.add_api_route(prefix + "/regulations", self.catalogue, methods=["GET"],
                                      include_in_schema=visible)
            self.router.add_api_route(prefix + "/regulations/{rule_id}", self.regulation, methods=["GET"],
                                      include_in_schema=visible)

    async def upload(self, request: Request) -> JSONResponse:
        session = request.state.browser_session
        manager: JobManager = request.app.state.jobs
        if request.headers.get("transfer-encoding") is not None:
            raise APIError(400, "unsupported_transfer_encoding", "Use Content-Length without Transfer-Encoding.")
        lengths = request.headers.getlist("content-length")
        if not lengths:
            raise APIError(411, "length_required", "Content-Length is required.")
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,12}", lengths[0]):
            raise APIError(400, "invalid_length", "Invalid Content-Length.")
        length = int(lengths[0])
        if length > MAX_UPLOAD_BYTES:
            raise APIError(413, "upload_too_large", "Multipart upload exceeds 64 MiB.")
        if length <= 0:
            raise APIError(400, "empty_upload", "The request body is empty.")
        if not manager.reserve():
            raise APIError(429, "queue_full", "The analysis queue is full. Try again later.")
        reserved = True
        deadline = request.state.received_at + JOB_TIMEOUT_SECONDS
        try:
            body = bytearray()
            stream = request.stream().__aiter__()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APIError(408, "upload_timeout", "Upload exceeded the analysis deadline.")
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), min(30, remaining))
                except StopAsyncIteration:
                    break
                if len(body) + len(chunk) > length:
                    raise APIError(400, "invalid_length", "Body exceeds Content-Length.")
                body.extend(chunk)
            if len(body) != length:
                raise APIError(400, "incomplete_upload", "The request body is incomplete.")
            payload, suffix = await run_in_threadpool(extract_video_upload, bytes(body), request.headers.get("content-type", ""))
            if time.monotonic() >= deadline:
                raise APIError(408, "upload_timeout", "Upload exceeded the analysis deadline.")
            job_id = uuid.uuid4().hex
            # The storage task owns the reservation and file even if the HTTP
            # request is cancelled; its finally block completes the handoff/cleanup.
            storage = asyncio.create_task(asyncio.to_thread(self._store_reserved, manager, payload, deadline, job_id, suffix))
            reserved = False
            try:
                await asyncio.shield(storage)
            except asyncio.CancelledError:
                # Observe a detached failure without logging paths or video data.
                storage.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                raise
            self._auth.remember(job_id, session)
            return JSONResponse({"job_id": job_id, "status": "processing"}, status_code=202)
        except UploadError as exc:
            raise APIError(exc.status, exc.code, exc.message) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise APIError(408, "upload_timeout", "Upload exceeded the analysis deadline.") from exc
        except ClientDisconnect as exc:
            raise APIError(400, "incomplete_upload", "The request body is incomplete.") from exc
        except (OSError, RuntimeError) as exc:
            raise APIError(503, "server_unavailable", "The local service cannot accept this video.") from exc
        finally:
            if reserved:
                manager.release()

    @staticmethod
    def _store_reserved(manager: JobManager, payload: bytes, deadline: float, job_id: str,
                        suffix: str = ".mov") -> None:
        reserved = True
        upload_path = None
        try:
            if time.monotonic() >= deadline:
                raise APIError(408, "upload_timeout", "Upload exceeded the analysis deadline.")
            upload_path = manager.runtime_dir / (job_id + suffix)
            with upload_path.open("xb") as target:
                target.write(payload)
            # submit_reserved always consumes the reservation, including failure.
            reserved = False
            manager.submit_reserved(upload_path, deadline, job_id)
            upload_path = None
        finally:
            if reserved:
                manager.release()
            if upload_path is not None:
                try:
                    upload_path.unlink(missing_ok=True)
                except OSError:
                    pass

    async def job(self, request: Request, job_id: str) -> dict:
        session = request.state.browser_session
        result = request.app.state.jobs.get(job_id) if _JOB_ID.fullmatch(job_id) else None
        if result is None or not self._auth.owns(job_id, session):
            raise APIError(404, "job_not_found", "Job not found or expired.")
        return result

    async def catalogue(self, request: Request) -> dict:
        return self._regulations.catalogue()

    async def regulation(self, request: Request, rule_id: str) -> dict:
        result = self._regulations.catalogue(rule_id) if _RULE_ID.fullmatch(rule_id) else None
        if result is None:
            raise APIError(404, "regulation_not_found", "Regulation not found.")
        return result


def create_application(
    config: dict, *, api_key: str, host: str = "127.0.0.1", port: int = 8001,
    analyzer: VideoAnalyzer | None = None,
) -> FastAPI:
    repository = DictionaryRegulationRepository()
    service = AnalysisService(analyzer if analyzer is not None else LocalAIAnalyzer(), repository)
    configuration = dict(config)
    configuration.setdefault("runtime_dir", str(ROOT / "ai_service/data/runtime"))
    frontend_root = Path(configuration.get("frontend_dir", ROOT / "frontend/dist")).resolve()
    auth = SessionAuthenticator(api_key, public_origin=os.environ.get("FORTE_PUBLIC_ORIGIN"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = JobManager(configuration, service)
        try:
            yield
        finally:
            app.state.jobs.close()

    app = FastAPI(title="ForteLens Backend", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        request.state.received_at = time.monotonic()
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer"})
        return response

    @app.exception_handler(APIError)
    async def api_error(_request: Request, exc: APIError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        return JSONResponse({"error": {"code": "invalid_request", "message": "Invalid request."}}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        return JSONResponse({"error": {"code": "not_found" if exc.status_code == 404 else "http_error",
                                       "message": "Endpoint or method unavailable."}}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _exc: Exception):
        return JSONResponse({"error": {"code": "server_unavailable",
                                       "message": "The local service is temporarily unavailable."}}, status_code=500)

    app.include_router(AnalysisController(repository, auth).router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "local", "backend_service": "forte_backend",
                "framework": "fastapi", "workers": 1, "ui_auth": "local_session", "ui": "fortelens",
                "lan": host == "0.0.0.0"}

    async def static_response(request: Request, path: str):
        content_type = "text/html; charset=utf-8" if path in ("/", "/index.html") else FRONTEND_ASSETS.get(path)
        if content_type is None:
            raise APIError(404, "not_found", "Endpoint not found.")
        location = (frontend_root / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        try:
            if not location.is_relative_to(frontend_root) or not location.is_file() or location.stat().st_size > MAX_STATIC_BYTES:
                raise OSError("Frontend asset unavailable")
            content = location.read_bytes() if location.name == "index.html" else None
        except OSError as exc:
            raise APIError(503, "web_unavailable", "The local web interface is unavailable.") from exc
        response = (Response(content, media_type=content_type) if content is not None
                    else FileResponse(location, media_type=content_type))
        response.headers["Content-Security-Policy"] = _CSP
        if content is not None:
            response.set_cookie(auth.cookie_name, auth.cookie(request), httponly=True,
                                samesite="strict", secure=auth.public_request(request), path="/ui/api/v1")
        return response

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        return await static_response(request, "/")

    @app.get("/{asset_path:path}", include_in_schema=False)
    async def asset(request: Request, asset_path: str):
        return await static_response(request, "/" + asset_path)

    return app
