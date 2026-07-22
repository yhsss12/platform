from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from urllib.parse import urlparse

try:
    from starlette.requests import Request
except Exception:  # pragma: no cover
    Request = object  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedPublicEndpoint:
    base_url: str
    scheme: str
    host: str
    port: int
    source: str
    is_public: bool
    resolved_at_ts: float
    signature: str = ""

    @property
    def computed_at(self) -> float:
        return self.resolved_at_ts


def _sanitize_token(v: str) -> str:
    s = (v or "").strip().strip("'").strip('"').strip("`").strip()
    if not s:
        return ""
    s = s.replace("`", "").replace(" ", "")
    return s


def _first_header_value(v: str) -> str:
    if not v:
        return ""
    return v.split(",")[0].strip()


def _parse_forwarded_header(forwarded: str) -> dict:
    s = (forwarded or "").strip()
    if not s:
        return {}
    first = _first_header_value(s)
    parts = [p.strip() for p in first.split(";") if p.strip()]
    out: dict = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kk = k.strip().lower()
        vv = v.strip().strip('"').strip("'")
        if kk in ("host", "proto"):
            out[kk] = vv
    return out


def _parse_host_and_port(host: str) -> tuple[str, Optional[int]]:
    h = (host or "").strip()
    if not h:
        return "", None
    if h.startswith("[") and "]" in h:
        host_part, rest = h.split("]", 1)
        host_part = host_part[1:]
        rest = rest.strip()
        if rest.startswith(":"):
            try:
                return host_part, int(rest[1:])
            except Exception:
                return host_part, None
        return host_part, None
    if ":" in h and h.count(":") > 1:
        return h, None
    if ":" in h:
        hp, pp = h.rsplit(":", 1)
        try:
            return hp, int(pp)
        except Exception:
            return h, None
    return h, None


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return bool(getattr(addr, "is_global", False))
    except Exception:
        return False


def _is_loopback_or_local(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return bool(addr.is_loopback or addr.is_link_local)
    except Exception:
        return False


def _normalize_base_url(base_url: str) -> str:
    s = _sanitize_token(base_url).rstrip("/")
    if not s:
        return ""
    u = urlparse(s)
    if u.scheme not in ("http", "https") or not u.hostname:
        return ""
    if u.port is None:
        port = 443 if u.scheme == "https" else 80
        if (u.scheme == "http" and port == 80) or (u.scheme == "https" and port == 443):
            return f"{u.scheme}://{u.hostname}"
        return f"{u.scheme}://{u.hostname}:{port}"
    if (u.scheme == "http" and u.port == 80) or (u.scheme == "https" and u.port == 443):
        return f"{u.scheme}://{u.hostname}"
    return f"{u.scheme}://{u.hostname}:{u.port}"


def _default_port_for_scheme(scheme: str) -> int:
    return 443 if (scheme or "").lower() == "https" else 80


def _udp_src_ip(host: str, port: int, family: int) -> str:
    sock = socket.socket(family, socket.SOCK_DGRAM)
    try:
        if family == socket.AF_INET6:
            sock.connect((host, port, 0, 0))
        else:
            sock.connect((host, port))
        return str(sock.getsockname()[0] or "")
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _ip_route_get(target: str, *, ipv6: bool) -> str:
    cmd = ["ip"]
    if ipv6:
        cmd.append("-6")
    cmd += ["route", "get", target]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=1.5).decode("utf-8", "replace")
        return out.strip()
    except Exception:
        return ""


def _parse_ip_route_src(text: str) -> str:
    m = re.search(r"\bsrc\s+([0-9a-fA-F:\.]+)\b", text or "")
    return m.group(1).strip() if m else ""


def _gather_ip_candidates() -> list[str]:
    out: list[str] = []
    try:
        ip4 = _udp_src_ip("1.1.1.1", 80, socket.AF_INET)
        if ip4:
            out.append(ip4)
    except Exception:
        pass
    try:
        ip6 = _udp_src_ip("2606:4700:4700::1111", 80, socket.AF_INET6)
        if ip6:
            out.append(ip6)
    except Exception:
        pass
    r4 = _parse_ip_route_src(_ip_route_get("1.1.1.1", ipv6=False))
    if r4:
        out.append(r4)
    r6 = _parse_ip_route_src(_ip_route_get("2606:4700:4700::1111", ipv6=True))
    if r6:
        out.append(r6)
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
        for fam, _, _, _, sockaddr in infos:
            if fam == socket.AF_INET and isinstance(sockaddr, tuple) and sockaddr:
                out.append(str(sockaddr[0] or ""))
            elif fam == socket.AF_INET6 and isinstance(sockaddr, tuple) and sockaddr:
                out.append(str(sockaddr[0] or ""))
    except Exception:
        pass
    uniq: list[str] = []
    seen = set()
    for ip in out:
        s = str(ip or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _signature_from_candidates(values: Iterable[str]) -> str:
    return "|".join(sorted({str(v or "").strip() for v in values if str(v or "").strip()}))


def _choose_best_host(candidates: Iterable[str]) -> tuple[str, bool, str]:
    ips = [c for c in candidates if c and not _is_loopback_or_local(c)]
    public_v4 = [ip for ip in ips if ":" not in ip and _is_public_ip(ip)]
    public_v6 = [ip for ip in ips if ":" in ip and _is_public_ip(ip)]
    private_v4 = [ip for ip in ips if ":" not in ip and not _is_public_ip(ip)]
    private_v6 = [ip for ip in ips if ":" in ip and not _is_public_ip(ip)]
    if public_v4:
        return public_v4[0], True, "system_ip_public_v4"
    if public_v6:
        return public_v6[0], True, "system_ip_public_v6"
    if private_v4:
        return private_v4[0], False, "system_ip_private_v4"
    if private_v6:
        return private_v6[0], False, "system_ip_private_v6"
    return "127.0.0.1", False, "fallback_loopback"


def _maybe_domain_for_public_ip(ip: str) -> str:
    try:
        name = socket.getfqdn(ip)
        s = (name or "").strip()
        if not s or s.lower() in ("localhost", "localhost.localdomain") or s == ip:
            return ""
        infos = socket.getaddrinfo(s, None)
        for fam, _, _, _, sockaddr in infos:
            if fam == socket.AF_INET and isinstance(sockaddr, tuple) and sockaddr:
                if _is_public_ip(str(sockaddr[0] or "")):
                    return s
            if fam == socket.AF_INET6 and isinstance(sockaddr, tuple) and sockaddr:
                if _is_public_ip(str(sockaddr[0] or "")):
                    return s
        return ""
    except Exception:
        return ""


def _fingerprint() -> str:
    host = ""
    try:
        host = socket.gethostname()
    except Exception:
        host = ""
    return "|".join([host] + sorted(_gather_ip_candidates()))


class PublicEndpointResolver:
    def __init__(
        self,
        *,
        ttl_sec: float = 30.0,
        env_base_url: str = "",
        default_base_url: str = "",
        ttl_seconds: Optional[int] = None,
        system_probe: Optional[Callable[[], list[str]]] = None,
    ) -> None:
        if ttl_seconds is not None:
            ttl_sec = float(ttl_seconds)
        self.ttl_sec = float(max(1.0, ttl_sec))
        self._fingerprint_check_sec = float(max(1.0, min(5.0, self.ttl_sec)))
        self.env_base_url = _normalize_base_url(env_base_url or os.environ.get("PUBLIC_BASE_URL", ""))
        self.default_base_url = _normalize_base_url(default_base_url) or "http://127.0.0.1:8000"
        self._system_probe = system_probe
        self._cached: Optional[ResolvedPublicEndpoint] = None
        self._cached_fingerprint: str = ""
        self._cached_checked_ts: float = 0.0

    def resolve(self, request: Optional[Request] = None, *, force_refresh: bool = False) -> ResolvedPublicEndpoint:
        out = self._resolve_from_request(request)
        if out is not None:
            return out
        return self._resolve_from_system(force_refresh=force_refresh)

    def refresh(self, request: Optional[Request] = None, *, force: bool = False) -> ResolvedPublicEndpoint:
        should_force = True if request is None else force
        return self.resolve(request=request, force_refresh=should_force)

    def get_base_url(self, request: Optional[Request] = None) -> str:
        return self.resolve(request=request).base_url

    def _resolve_from_request(self, request: Optional[Request]) -> Optional[ResolvedPublicEndpoint]:
        if request is None or not hasattr(request, "headers"):
            return None
        try:
            hdr = request.headers
            forwarded = _parse_forwarded_header(_sanitize_token(str(hdr.get("forwarded") or "")))
            xf_host = _sanitize_token(_first_header_value(str(hdr.get("x-forwarded-host") or "")))
            xf_proto = _sanitize_token(_first_header_value(str(hdr.get("x-forwarded-proto") or "")))
            xf_port = _sanitize_token(_first_header_value(str(hdr.get("x-forwarded-port") or "")))
            raw_host = xf_host or forwarded.get("host") or _sanitize_token(str(hdr.get("host") or ""))
            raw_proto = (
                xf_proto
                or forwarded.get("proto")
                or _sanitize_token(str(getattr(getattr(request, "url", None), "scheme", "") or ""))
                or "http"
            )
            raw_port = xf_port
            if not raw_host:
                return None
            host, port_from_host = _parse_host_and_port(raw_host)
            if not host:
                return None
            scheme = "https" if raw_proto.lower() == "https" else "http"
            if port_from_host is not None:
                port = int(port_from_host)
            elif raw_port.strip().isdigit():
                port = int(raw_port.strip())
            else:
                try:
                    req_port = int(getattr(getattr(request, "url", None), "port", None) or 0)
                    port = req_port or _default_port_for_scheme(scheme)
                except Exception:
                    port = _default_port_for_scheme(scheme)
            if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
                base = f"{scheme}://{host}"
            else:
                base = f"{scheme}://{host}:{port}"
            return ResolvedPublicEndpoint(
                base_url=base,
                scheme=scheme,
                host=host,
                port=port,
                source="request_headers",
                is_public=_is_public_ip(host),
                resolved_at_ts=time.time(),
                signature=f"request:{base}",
            )
        except Exception:
            return None

    def _resolve_from_system(self, *, force_refresh: bool) -> ResolvedPublicEndpoint:
        now = time.time()
        env = _normalize_base_url(self.env_base_url or os.environ.get("PUBLIC_BASE_URL", ""))
        if env != self.env_base_url:
            self.env_base_url = env

        candidates = self._system_probe() if self._system_probe is not None else _gather_ip_candidates()
        fp = _signature_from_candidates(candidates) if self._system_probe is not None else _fingerprint()

        if not force_refresh and self._cached is not None:
            if (now - self._cached_checked_ts) < self._fingerprint_check_sec:
                return self._cached
            self._cached_checked_ts = now
            if fp and fp == self._cached_fingerprint and (now - self._cached.resolved_at_ts) < self.ttl_sec:
                return self._cached

        try:
            port_hint = 8000
            dflt = _normalize_base_url(self.default_base_url) or ""
            if dflt:
                parsed = urlparse(dflt)
                if parsed.port is not None:
                    port_hint = int(parsed.port)
                elif parsed.scheme:
                    port_hint = _default_port_for_scheme(parsed.scheme)

            host_ip, is_public, src = _choose_best_host(candidates)
            host_domain = _maybe_domain_for_public_ip(host_ip) if is_public else ""
            host = host_domain or host_ip
            scheme = "http"
            port = port_hint
            if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
                base_auto = f"{scheme}://{host}"
            else:
                base_auto = f"{scheme}://{host}:{port}"
            auto = ResolvedPublicEndpoint(
                base_url=base_auto,
                scheme=scheme,
                host=host,
                port=port,
                source=src if not host_domain else "system_dns_domain",
                is_public=is_public,
                resolved_at_ts=now,
                signature=fp or f"auto:{base_auto}",
            )
            fallback = self._resolve_env_or_default(now)
            if env and (auto.host in ("127.0.0.1", "localhost", "::1") or auto.source == "fallback_loopback"):
                chosen = fallback
            else:
                chosen = auto if auto.base_url else fallback
        except Exception:
            chosen = self._resolve_env_or_default(now)

        if force_refresh or self._cached is None or chosen.base_url != self._cached.base_url:
            logger.info("public endpoint resolved: %s (source=%s)", chosen.base_url, chosen.source)
        else:
            logger.debug("public endpoint cached: %s (source=%s)", chosen.base_url, chosen.source)

        self._cached = chosen
        self._cached_fingerprint = fp or self._cached_fingerprint
        self._cached_checked_ts = now
        return chosen

    def _resolve_env_or_default(self, now: float) -> ResolvedPublicEndpoint:
        env = self.env_base_url
        if env:
            parsed = urlparse(env)
            scheme = (parsed.scheme or "http").lower()
            host = parsed.hostname or "127.0.0.1"
            port = int(parsed.port) if parsed.port is not None else _default_port_for_scheme(scheme)
            return ResolvedPublicEndpoint(
                base_url=env,
                scheme=scheme,
                host=host,
                port=port,
                source="env_public_base_url",
                is_public=_is_public_ip(host),
                resolved_at_ts=now,
                signature=f"env:{env}",
            )
        dflt = _normalize_base_url(self.default_base_url) or "http://127.0.0.1:8000"
        parsed = urlparse(dflt)
        scheme = (parsed.scheme or "http").lower()
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port) if parsed.port is not None else _default_port_for_scheme(scheme)
        return ResolvedPublicEndpoint(
            base_url=dflt,
            scheme=scheme,
            host=host,
            port=port,
            source="default_public_base_url",
            is_public=_is_public_ip(host),
            resolved_at_ts=now,
            signature=f"default:{dflt}",
        )


public_endpoint_resolver = PublicEndpointResolver()
_global_resolver: Optional[PublicEndpointResolver] = public_endpoint_resolver


def get_public_endpoint_resolver(*, default_base_url: str = "") -> PublicEndpointResolver:
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = PublicEndpointResolver(default_base_url=default_base_url)
    if default_base_url:
        _global_resolver.default_base_url = _normalize_base_url(default_base_url) or _global_resolver.default_base_url
    env = _normalize_base_url(os.environ.get("PUBLIC_BASE_URL", ""))
    if env:
        _global_resolver.env_base_url = env
    return _global_resolver
