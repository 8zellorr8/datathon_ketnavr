"""API-key authentication and signed, same-origin local browser sessions."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import time
from urllib.parse import urlsplit

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from .http_errors import APIError

_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SessionAuthenticator:
    cookie_name = "fortelens_session"

    def __init__(self, api_key: str):
        if not isinstance(api_key, str) or len(api_key) < 16:
            raise ValueError("API key must contain at least 16 characters")
        self._api_key = api_key.encode("utf-8")
        self._secret = secrets.token_bytes(32)
        self._jobs: dict[str, tuple[str, float]] = {}

    def _session_id(self, request: Request) -> str | None:
        cookie = request.cookies.get(self.cookie_name, "")
        if not re.fullmatch(r"[0-9a-f]{48}\.[0-9a-f]{64}", cookie):
            return None
        identifier, signature = cookie.split(".")
        expected = hmac.new(self._secret, identifier.encode(), hashlib.sha256).hexdigest()
        return identifier if hmac.compare_digest(signature, expected) else None

    def cookie(self, request: Request) -> str:
        identifier = self._session_id(request) or secrets.token_hex(24)
        signature = hmac.new(self._secret, identifier.encode(), hashlib.sha256).hexdigest()
        return identifier + "." + signature

    async def authorize(self, request: Request, key: str | None = Security(_key_header)) -> str | None:
        if not request.url.path.startswith("/ui/api/v1/"):
            provided = key.encode("utf-8") if key and len(request.headers.getlist("x-api-key")) == 1 else b""
            if not hmac.compare_digest(provided, self._api_key):
                raise APIError(401, "unauthorized", "A valid X-API-Key is required.")
            request.state.browser_session = None
            return None
        host = request.headers.get("host", "")
        try:
            parsed = urlsplit("http://" + host)
            name = parsed.hostname
            local_host = name == "localhost" or (
                ipaddress.ip_address(name).is_private and not ipaddress.ip_address(name).is_unspecified
            )
            port = request.scope.get("server", (None, None))[1]
            valid_host = local_host and parsed.port == port and not parsed.username and not parsed.path
        except (ValueError, TypeError):
            valid_host = False
        origin = request.headers.get("origin")
        valid_origin = origin is None or origin == "http://" + host
        if request.method == "POST":
            valid_origin = origin == "http://" + host
        same_site = request.headers.get("sec-fetch-site", "same-origin") not in ("cross-site", "same-site")
        session = self._session_id(request)
        if not (valid_host and valid_origin and same_site and session):
            raise APIError(403, "unauthorized", "Reload the local website to start a browser session.")
        request.state.browser_session = session
        return session

    def remember(self, job_id: str, session: str | None) -> None:
        now = time.monotonic()
        self._jobs = {key: item for key, item in self._jobs.items() if now - item[1] < 3900}
        if session:
            while len(self._jobs) >= 1024:
                del self._jobs[next(iter(self._jobs))]
            self._jobs[job_id] = (session, now)

    def owns(self, job_id: str, session: str | None) -> bool:
        return session is None or self._jobs.get(job_id, (None, 0))[0] == session
