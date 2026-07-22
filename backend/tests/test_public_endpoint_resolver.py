import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.requests import Request

from app.services.public_endpoint_resolver import PublicEndpointResolver


def _make_request(headers: dict, *, scheme: str = "http", server=("testserver", 8000)) -> Request:
    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": scheme,
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(k.lower().encode("utf-8"), str(v).encode("utf-8")) for k, v in headers.items()],
        "client": ("127.0.0.1", 12345),
        "server": server,
        "root_path": "",
        "extensions": {},
    }
    return Request(scope)


class TestPublicEndpointResolver(unittest.TestCase):
    def test_request_forwarded_headers_take_priority(self):
        req = _make_request(
            {
                "x-forwarded-host": "api.example.com",
                "x-forwarded-proto": "https",
                "x-forwarded-port": "443",
            },
            scheme="http",
            server=("internal", 8000),
        )
        resolver = PublicEndpointResolver(ttl_sec=30, env_base_url="", default_base_url="http://127.0.0.1:8000")
        out = resolver.resolve(req)
        self.assertEqual(out.base_url, "https://api.example.com")
        self.assertEqual(out.scheme, "https")
        self.assertEqual(out.host, "api.example.com")
        self.assertEqual(out.port, 443)
        self.assertEqual(out.source, "request_headers")

    def test_request_host_header_with_port(self):
        req = _make_request({"host": "svc.local:8443"}, scheme="http", server=("svc.local", 8000))
        resolver = PublicEndpointResolver(ttl_sec=30, env_base_url="", default_base_url="http://127.0.0.1:8000")
        out = resolver.resolve(req)
        self.assertEqual(out.base_url, "http://svc.local:8443")
        self.assertEqual(out.port, 8443)

    def test_forwarded_header_supported(self):
        req = _make_request({"forwarded": "proto=https;host=proxy.example.com:444"}, scheme="http", server=("internal", 8000))
        resolver = PublicEndpointResolver(ttl_sec=30, env_base_url="", default_base_url="http://127.0.0.1:8000")
        out = resolver.resolve(req)
        self.assertEqual(out.base_url, "https://proxy.example.com:444")
        self.assertEqual(out.scheme, "https")
        self.assertEqual(out.port, 444)

    def test_system_prefers_public_ip_over_private(self):
        resolver = PublicEndpointResolver(ttl_sec=30, env_base_url="", default_base_url="http://127.0.0.1:8000")
        with patch("app.services.public_endpoint_resolver._gather_ip_candidates", return_value=["10.0.0.2", "8.8.8.8"]), patch(
            "app.services.public_endpoint_resolver._maybe_domain_for_public_ip", return_value=""
        ), patch("app.services.public_endpoint_resolver._fingerprint", return_value="fp1"), patch(
            "app.services.public_endpoint_resolver.time.time", return_value=1000.0
        ):
            out = resolver.resolve(None, force_refresh=True)
        self.assertEqual(out.host, "8.8.8.8")
        self.assertTrue(out.is_public)

    def test_env_fallback_when_only_loopback_detected(self):
        resolver = PublicEndpointResolver(ttl_sec=30, env_base_url="http://env.example.com:8000", default_base_url="http://127.0.0.1:8000")
        with patch("app.services.public_endpoint_resolver._gather_ip_candidates", return_value=["127.0.0.1"]), patch(
            "app.services.public_endpoint_resolver._fingerprint", return_value="fp2"
        ), patch("app.services.public_endpoint_resolver.time.time", return_value=2000.0):
            out = resolver.resolve(None, force_refresh=True)
        self.assertEqual(out.base_url, "http://env.example.com:8000")
        self.assertEqual(out.source, "env_public_base_url")

    def test_ipv6_public_selected_when_no_public_ipv4(self):
        resolver = PublicEndpointResolver(ttl_sec=30, env_base_url="", default_base_url="http://127.0.0.1:8000")
        with patch("app.services.public_endpoint_resolver._gather_ip_candidates", return_value=["fd00::1", "2001:4860:4860::8888"]), patch(
            "app.services.public_endpoint_resolver._maybe_domain_for_public_ip", return_value=""
        ), patch("app.services.public_endpoint_resolver._fingerprint", return_value="fp6"), patch(
            "app.services.public_endpoint_resolver.time.time", return_value=2500.0
        ):
            out = resolver.resolve(None, force_refresh=True)
        self.assertEqual(out.host, "2001:4860:4860::8888")
        self.assertTrue(out.is_public)

    def test_dynamic_refresh_on_fingerprint_change(self):
        resolver = PublicEndpointResolver(ttl_sec=60, env_base_url="", default_base_url="http://127.0.0.1:8000")
        with patch("app.services.public_endpoint_resolver._gather_ip_candidates", return_value=["10.0.0.2"]), patch(
            "app.services.public_endpoint_resolver._fingerprint", side_effect=["fpA", "fpB"]
        ), patch("app.services.public_endpoint_resolver.time.time", side_effect=[3000.0, 3006.0]):
            out1 = resolver.resolve(None, force_refresh=True)
            out2 = resolver.resolve(None)
        self.assertNotEqual(out1.resolved_at_ts, out2.resolved_at_ts)

    def test_compat_get_base_url_and_refresh(self):
        resolver = PublicEndpointResolver(ttl_seconds=300, system_probe=lambda: ["192.168.1.10", "8.8.8.8"])
        self.assertEqual(resolver.get_base_url(), "http://8.8.8.8:8000")
        refreshed = resolver.refresh(force=True)
        self.assertEqual(refreshed.base_url, "http://8.8.8.8:8000")


if __name__ == "__main__":
    unittest.main()
