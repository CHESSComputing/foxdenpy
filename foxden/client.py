"""Low level HTTP client for talking to FOXDEN services.

FOXDEN services all respond with a common JSON envelope, e.g.:

    {
      "http_code": 200,
      "service_code": 0,
      "service": "DataDiscovery",
      "error": "",
      "results": {"nrecords": 1, "records": [...]},
      ...
    }

`FoxdenClient` centralizes auth headers, retries, and unwraps/validates
that envelope so the rest of the package can just deal in plain dicts.

Enable request-level debug logging with:

    import logging
    logging.basicConfig(level=logging.DEBUG)
"""
from __future__ import annotations

import logging
import threading
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import FoxdenConfig
from .exceptions import AuthenticationError, FoxdenError, ServiceError

logger = logging.getLogger("foxden")


class FoxdenClient:
    def __init__(self, config: FoxdenConfig):
        self.config = config
        # requests.Session is not guaranteed thread-safe for concurrent
        # use of its underlying connection pool -- give each thread
        # (e.g. each worker downloading a file in a training_data()
        # batch) its own Session rather than sharing one, to avoid
        # contention/hangs under concurrent access.
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            retries = Retry(
                total=self.config.retries,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET", "HEAD"),
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _headers(self, accept: str = "application/json") -> Dict[str, str]:
        headers = {"Accept": accept}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return headers

    def _timeout(self) -> Tuple[float, float]:
        # (connect_timeout, read_timeout): each phase of a request is
        # bounded independently, so a server that accepts a connection
        # but never responds can't hang past `read_timeout`.
        return (self.config.connect_timeout, self.config.timeout)

    def get_json(
        self, base_url: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = base_url.rstrip("/") + path
        logger.debug("GET %s params=%s", url, params)
        resp = self._session().get(
            url,
            params=params,
            headers=self._headers("application/json"),
            timeout=self._timeout(),
            verify=self.config.verify_ssl,
        )
        self._raise_for_status(resp)
        data = resp.json()
        self._check_envelope(data)
        return data

    def get_raw(
        self, base_url: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """GET and return parsed JSON as-is, with no envelope unwrapping
        or validation. Used for endpoints like `/dm` that respond with a
        plain JSON array rather than FOXDEN's usual
        `{http_code, service_code, results, ...}` envelope.
        """
        url = base_url.rstrip("/") + path
        logger.debug("GET %s params=%s", url, params)
        resp = self._session().get(
            url,
            params=params,
            headers=self._headers("application/json"),
            timeout=self._timeout(),
            verify=self.config.verify_ssl,
        )
        self._raise_for_status(resp)
        return resp.json()

    def get_bytes(
        self, base_url: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> bytes:
        url = base_url.rstrip("/") + path
        logger.debug("GET %s params=%s", url, params)
        resp = self._session().get(
            url,
            params=params,
            headers=self._headers("application/octet-stream"),
            timeout=self._timeout(),
            verify=self.config.verify_ssl,
        )
        self._raise_for_status(resp)
        return resp.content

    def stream(
        self,
        base_url: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[bytes]:
        """Yield an object's content in chunks without loading it fully
        into memory."""
        url = base_url.rstrip("/") + path
        logger.debug("GET %s params=%s (stream)", url, params)
        resp = self._session().get(
            url,
            params=params,
            headers=self._headers("application/octet-stream"),
            timeout=self._timeout(),
            verify=self.config.verify_ssl,
            stream=True,
        )
        self._raise_for_status(resp)
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code in (401, 403):
            raise AuthenticationError(
                f"authentication failed ({resp.status_code}) for {resp.url}; "
                "set FOXDEN_TOKEN, ~/.foxden/token, or pass token= explicitly"
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise FoxdenError(f"HTTP error from {resp.url}: {exc}") from exc

    @staticmethod
    def _check_envelope(data: Dict[str, Any]) -> None:
        error = data.get("error")
        service_code = data.get("service_code", 0)
        if error or service_code:
            raise ServiceError(
                f"{data.get('service', 'FOXDEN')} service error: {error or service_code}",
                service=data.get("service", ""),
                service_code=service_code,
            )


def encode_did(did: str) -> str:
    """Percent-encode a FOXDEN DID for manual URL construction.

    DIDs are path-style identifiers, e.g.
    '/beamline=3a/btr=test-demo/cycle=2026-1/sample_name=test-demo/test=composite'
    and are fully percent-encoded (including '/') as the `did` query
    parameter value, matching FOXDEN's documented curl usage.

    Not needed for normal use of this package -- `requests` already
    encodes query parameters passed via `params=` the same way -- but
    exposed for callers building URLs by hand.
    """
    return urllib.parse.quote(did, safe="")
