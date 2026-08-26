"""foxden: a lightweight Python client for the FOXDEN data ecosystem.

    import foxden

    ds = foxden.open("/beamline=3a/btr=shanks-4838-a/cycle=2026-1/sample_name=ti7-19-tomo")

    print(ds.metadata)                 # GET /record?did=...

    print(ds.tree().pretty())          # full nested tree, via /dm
    data = ds.get_file("5/meta/1", "scalars.txt")   # GET /dm?did=...&path=...&file=...

    for batch in ds.training_data(batch_size=128):
        ...

Configuration (frontend URL + auth token) comes from, in order of
precedence: explicit arguments -> environment variables
(FOXDEN_URL / FOXDEN_TOKEN) -> ~/.foxden.yaml -> ~/.foxden/token ->
built-in defaults. See `foxden.config.FoxdenConfig` for details.

For visibility into what's happening on slow/large datasets, enable
debug logging:

    import logging
    logging.basicConfig(level=logging.DEBUG)
"""
from .config import FoxdenConfig
from .dataset import Dataset
from .dm import DMClient, DMEntry, DMNode, resolve_fetch_path
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    FoxdenError,
    RecordNotFoundError,
    ServiceError,
)

__all__ = [
    "open",
    "Dataset",
    "DMEntry",
    "DMNode",
    "DMClient",
    "resolve_fetch_path",
    "FoxdenConfig",
    "FoxdenError",
    "ConfigurationError",
    "AuthenticationError",
    "RecordNotFoundError",
    "ServiceError",
]

__version__ = "0.2.1"


def open(
    did: str,
    url: str | None = None,
    token: str | None = None,
    config: FoxdenConfig | None = None,
) -> Dataset:
    """Open a FOXDEN dataset by its DID and return a `Dataset` handle.

    Nothing is fetched over the network until `.metadata`, `.tree()`,
    `.list()`, `.files()`, or `.training_data()` is first accessed.

    Args:
        did: FOXDEN dataset identifier, e.g.
            "/beamline=3a/btr=shanks-4838-a/cycle=2026-1/sample_name=ti7-19-tomo".
        url: override the FOXDEN frontend URL (else FOXDEN_URL env var,
            config file, or the built-in default). Used for both
            `/record` and `/dm`.
        token: override the bearer token (else FOXDEN_TOKEN env var,
            ~/.foxden/token file, or config file).
        config: a pre-built FoxdenConfig; takes precedence over url/token.
    """
    cfg = config or FoxdenConfig.load(url=url, token=token)
    return Dataset(did, config=cfg)
