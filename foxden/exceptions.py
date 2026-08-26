"""Exceptions raised by the foxden client."""


class FoxdenError(Exception):
    """Base exception for all foxden errors."""


class ConfigurationError(FoxdenError):
    """Raised when the client is misconfigured (e.g. no auth token)."""


class AuthenticationError(FoxdenError):
    """Raised on 401/403 responses from a FOXDEN service."""


class RecordNotFoundError(FoxdenError):
    """Raised when a DID does not resolve to a metadata record."""


class ServiceError(FoxdenError):
    """Raised when a FOXDEN service reports a service-level error
    (non-zero `service_code` / non-empty `error` field) even though the
    HTTP call itself succeeded.
    """

    def __init__(self, message: str, service: str = "", service_code: int = 0):
        super().__init__(message)
        self.service = service
        self.service_code = service_code
