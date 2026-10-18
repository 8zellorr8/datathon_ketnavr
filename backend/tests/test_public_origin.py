"""Pure request authentication contracts: no server, model or video is opened."""
import asyncio
import unittest

from starlette.requests import Request

from backend.auth import SessionAuthenticator
from backend.http_errors import APIError


class PublicOriginTests(unittest.TestCase):
    def setUp(self):
        self.auth = SessionAuthenticator("synthetic-test-key-123456", "https://demo.trycloudflare.com")

    def request(self, *, host="demo.trycloudflare.com", origin="https://demo.trycloudflare.com",
                method="POST", path="/ui/api/v1/analyze", session=True, extra=()):
        headers = [(b"host", host.encode())]
        if origin is not None:
            headers.append((b"origin", origin.encode()))
        headers.extend(extra)
        scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
                 "scheme": "http", "server": ("127.0.0.1", 8001), "headers": headers,
                 "query_string": b""}
        request = Request(scope)
        if session:
            cookie = self.auth.cookie(request)
            scope["headers"].append((b"cookie", (self.auth.cookie_name + "=" + cookie).encode()))
            request = Request(scope)
        return request

    def authorize(self, request, key=None):
        return asyncio.run(self.auth.authorize(request, key))

    def test_exact_public_https_origin(self):
        self.assertIsNotNone(self.authorize(self.request()))

    def test_public_cookie_is_secure_only_for_configured_host(self):
        self.assertTrue(self.auth.public_request(self.request()))
        self.assertFalse(self.auth.public_request(self.request(host="127.0.0.1:8001")))

    def test_other_public_host_rejected(self):
        with self.assertRaises(APIError):
            self.authorize(self.request(host="other.trycloudflare.com", origin="https://other.trycloudflare.com"))

    def test_wrong_origin_rejected(self):
        for origin in (None, "http://demo.trycloudflare.com", "https://evil.example", "null"):
            with self.subTest(origin=origin), self.assertRaises(APIError):
                self.authorize(self.request(origin=origin))

    def test_signed_session_required(self):
        with self.assertRaises(APIError):
            self.authorize(self.request(session=False))

    def test_local_http_still_works(self):
        self.assertIsNotNone(self.authorize(self.request(host="127.0.0.1:8001", origin="http://127.0.0.1:8001")))

    def test_forwarded_host_is_not_trusted(self):
        with self.assertRaises(APIError):
            self.authorize(self.request(host="evil.example", extra=[(b"x-forwarded-host", b"demo.trycloudflare.com")]))

    def test_cross_site_rejected(self):
        with self.assertRaises(APIError):
            self.authorize(self.request(extra=[(b"sec-fetch-site", b"cross-site")]))

    def test_duplicate_headers_rejected(self):
        for header in ((b"host", b"demo.trycloudflare.com"), (b"origin", b"https://demo.trycloudflare.com")):
            with self.subTest(header=header), self.assertRaises(APIError):
                self.authorize(self.request(extra=[header]))

    def test_public_api_still_requires_key(self):
        with self.assertRaises(APIError):
            self.authorize(self.request(path="/api/v1/analyze"))
        request = self.request(path="/api/v1/analyze", extra=[(b"x-api-key", b"synthetic-test-key-123456")])
        self.assertIsNone(self.authorize(request, "synthetic-test-key-123456"))

    def test_malformed_origin_config_rejected(self):
        for origin in ("http://demo.example", "https://*.example", "https://a.example/", "https://a.example/path",
                       "https://a.example?", "https://a.example#", "https://user@a.example", "https://a.example:99999",
                       "https://a..example", "https://a_example", "https://a.example\\evil", "https://A.example"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                SessionAuthenticator("synthetic-test-key-123456", origin)

    def test_no_public_origin_by_default(self):
        auth = SessionAuthenticator("synthetic-test-key-123456")
        with self.assertRaises(APIError):
            asyncio.run(auth.authorize(self.request()))


if __name__ == "__main__":
    unittest.main()
