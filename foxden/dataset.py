"""High level Dataset object returned by foxden.open()."""
from __future__ import annotations

import concurrent.futures
import fnmatch
import logging
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .client import FoxdenClient
from .config import FoxdenConfig
from .dm import DMClient, DMEntry, DMNode, resolve_fetch_path
from .exceptions import RecordNotFoundError

logger = logging.getLogger("foxden")

# A loader turns (DMEntry, raw_bytes) into whatever representation the
# caller wants (e.g. decode an image, parse a numpy array, ...).
Loader = Callable[[DMEntry, bytes], Any]


def _default_loader(entry: DMEntry, data: bytes) -> Dict[str, Any]:
    return {"path": entry.path, "name": entry.name, "data": data}


def _include_ok(name: str, include: Optional[Sequence[str]]) -> bool:
    """True if `name` should be kept per `include` patterns (no
    restriction -> always kept)."""
    if not include:
        return True
    return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in include)


def _exclude_ok(name: str, exclude: Optional[Sequence[str]]) -> bool:
    """True if `name` should be kept per `exclude` patterns (no
    patterns -> nothing excluded, always kept)."""
    if not exclude:
        return True
    return not any(fnmatch.fnmatch(name.lower(), p.lower()) for p in exclude)


class Dataset:
    """Represents a single FOXDEN dataset identified by its DID.

    Metadata is fetched lazily on first access to `.metadata` (or
    anything that depends on it) and cached; the directory tree is
    similarly fetched lazily on first access to `.tree()` / `.files()`
    / `.training_data()` and cached. Call `.refresh()` to force a
    re-fetch of either.
    """

    def __init__(self, did: str, config: Optional[FoxdenConfig] = None):
        self.did = did
        self.config = config or FoxdenConfig.load()
        self._http = FoxdenClient(self.config)
        self._dm = DMClient(self._http, self.config.url)
        self._metadata: Optional[Dict[str, Any]] = None
        self._tree: Optional[DMNode] = None

    # -- metadata ----------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """The dataset's metadata record, fetched from
        `GET /record?did=<did>` and cached after the first call."""
        if self._metadata is None:
            self._metadata = self._fetch_metadata()
        return self._metadata

    def _fetch_metadata(self) -> Dict[str, Any]:
        # requests percent-encodes params (including '/' -> %2F) the same
        # way FOXDEN's documented curl examples do, so `self.did` is
        # passed through as-is here.
        data = self._http.get_json(self.config.url, "/record", params={"did": self.did})
        records = (data.get("results") or {}).get("records") or []
        if not records:
            raise RecordNotFoundError(f"no metadata record found for did={self.did!r}")
        return records[0]

    def refresh(self) -> "Dataset":
        """Drop cached metadata/tree so the next access re-fetches them."""
        self._metadata = None
        self._tree = None
        return self

    # -- directory tree (/dm) -----------------------------------------

    def list(self, path: Optional[str] = None) -> List[DMEntry]:
        """Shallow (one-level) listing of entries under `path` (dataset
        root if `path` is None). Always a fresh call to `/dm`; does not
        use or populate the cached tree."""
        return self._dm.list_dir(self.did, path)

    def tree(
        self,
        refresh: bool = False,
        max_depth: Optional[int] = None,
        progress_every: int = 25,
    ) -> DMNode:
        """The full nested directory tree for this dataset, built by
        recursively walking `/dm` (one request per directory) and
        cached after the first call.

        Pass `refresh=True` to force re-fetching. `max_depth` limits how
        many levels deep to recurse (unlimited by default).

        Large trees take a while to walk -- an INFO-level progress line
        is logged every `progress_every` directories (0 to disable).
        Enable visibility with:

            import logging
            logging.basicConfig(level=logging.INFO)
        """
        if self._tree is None or refresh:
            self._tree = self._dm.tree(
                self.did, path=None, max_depth=max_depth, progress_every=progress_every
            )
        return self._tree

    def files(
        self,
        path: Optional[str] = None,
        max_depth: Optional[int] = None,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[DMEntry]:
        """All files (recursively) under `path` (the whole dataset if
        `path` is None). Uses the cached tree when possible.

        `include`/`exclude` are lists of case-insensitive glob patterns
        matched against each file's name, e.g. `include=["*.txt",
        "*.mcs"]` or `exclude=["*.h5"]`, to scope out large or
        irrelevant files before anything is downloaded.
        """
        if path is None:
            node = self.tree(max_depth=max_depth)
        else:
            cached = self._tree.find(path) if self._tree is not None else None
            if cached is not None and max_depth is None:
                node = cached
            else:
                node = self._dm.tree(self.did, path=path, max_depth=max_depth)
        entries = [n.entry for n in node.files()]
        if include or exclude:
            entries = [
                e for e in entries
                if _include_ok(e.name, include) and _exclude_ok(e.name, exclude)
            ]
        return entries

    def get_file(self, path: str, file: str) -> bytes:
        """Fetch a single file's raw bytes given an explicit directory
        `path` and `file` name, e.g.
        `ds.get_file("5/meta/1", "scalars.txt")`. Use this when you're
        specifying the location yourself; use `.fetch(entry)` when you
        have a `DMEntry` from `.list()` / `.files()` / `.tree()`.
        """
        return self._dm.get_file(self.did, path, file)

    def fetch(self, entry: DMEntry) -> bytes:
        """Fetch the raw bytes for a file `DMEntry` obtained from
        `.list()`, `.files()`, or `.tree()`. Correctly derives the
        `path=`/`file=` pair regardless of whether the server reports a
        file entry's `path` as its parent directory or as the full path
        including the filename (see `resolve_fetch_path`)."""
        path, file = resolve_fetch_path(entry)
        return self._dm.get_file(self.did, path, file)

    def stream(self, entry: DMEntry, chunk_size: int = 1 << 20):
        """Stream a file's content in chunks instead of loading it
        fully into memory -- useful for large files (e.g. multi-hundred
        MB `.h5` frames) where `.fetch()`'s full in-memory download
        isn't desirable."""
        path, file = resolve_fetch_path(entry)
        return self._dm.stream_file(self.did, path, file, chunk_size=chunk_size)

    # -- batch data access ---------------------------------------------

    def training_data(
        self,
        path: Optional[str] = None,
        batch_size: int = 128,
        loader: Optional[Loader] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        max_workers: int = 8,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        progress: bool = True,
    ) -> Iterator[List[Any]]:
        """Yield batches of this dataset's files for training/processing.

        Recursively collects every file under `path` (the whole dataset
        if omitted) -- optionally narrowed with `include`/`exclude`
        glob patterns against filenames, e.g. `exclude=["*.h5"]` to
        skip large raw frames -- then yields batches of at most
        `batch_size` items. Files within a batch are downloaded in
        parallel via a thread pool (`max_workers`).

        By default each item is `{"path": ..., "name": ..., "data":
        <bytes>}`; pass `loader` to transform `(DMEntry, bytes)` into
        your own representation, e.g.:

            def load_npy(entry, data):
                import numpy as np, io
                return np.load(io.BytesIO(data))

            for batch in ds.training_data(batch_size=64, loader=load_npy):
                ...

        Set `shuffle=True` to shuffle file order once up front (not
        re-shuffled between epochs -- call again for a new order).

        With `progress=True` (default), logs an INFO line after each
        batch (files/bytes so far, out of the total) so a long run is
        visibly making progress rather than looking stuck. Enable
        visibility with `logging.basicConfig(level=logging.INFO)`.
        """
        loader = loader or _default_loader
        entries = self.files(path, include=include, exclude=exclude)
        if not entries:
            return

        if shuffle:
            entries = entries[:]
            random.Random(seed).shuffle(entries)

        total = len(entries)
        done = 0
        total_bytes = 0
        for start in range(0, total, batch_size):
            chunk = entries[start : start + batch_size]
            batch, batch_bytes = self._load_batch(chunk, loader, max_workers)
            done += len(chunk)
            total_bytes += batch_bytes
            if progress:
                logger.info(
                    "foxden: training_data progress -- %d/%d files, %.1f MB downloaded",
                    done, total, total_bytes / (1 << 20),
                )
            yield batch

    def _load_batch(
        self, entries: List[DMEntry], loader: Loader, max_workers: int
    ):
        sizes: Dict[int, int] = {}

        def fetch(idx: int, entry: DMEntry) -> Any:
            path, file = resolve_fetch_path(entry)
            data = self._dm.get_file(self.did, path, file)
            sizes[idx] = len(data)
            return loader(entry, data)

        if max_workers <= 1 or len(entries) == 1:
            results = [fetch(i, e) for i, e in enumerate(entries)]
        else:
            results = [None] * len(entries)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_idx = {pool.submit(fetch, i, e): i for i, e in enumerate(entries)}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
        return results, sum(sizes.values())

    def __repr__(self) -> str:
        return f"Dataset(did={self.did!r})"
