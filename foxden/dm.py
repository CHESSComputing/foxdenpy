"""Client for FOXDEN's `/dm` data management endpoint.

The `/dm` endpoint browses a dataset's file tree and fetches individual
files:

    GET /dm?did=<did>                          -> list entries at the dataset root
    GET /dm?did=<did>&path=<path>              -> list entries under <path>
    GET /dm?did=<did>&path=<path>&file=<file>  -> fetch a file's raw bytes

Listing calls return a plain JSON array of entries (not FOXDEN's usual
`{http_code, results, ...}` envelope), e.g.:

    [
      {"did": "...", "esc_did": "...", "name": "1", "is_dir": true, "path": "1"},
      {"did": "...", "esc_did": "...", "name": "2", "is_dir": true, "path": "2"}
    ]

A directory entry's `path` is what you pass back in as `path=` to list
its contents (and to recurse further). A file entry's `path` is the
*directory* it lives in, and its `name` is the filename -- together
they're exactly the `path=`/`file=` pair `/dm` expects to fetch that
file, matching FOXDEN's documented example:
`/dm?did=$did&path=/5/meta/1&file=scalars.txt`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .client import FoxdenClient

logger = logging.getLogger("foxden")

# Hard ceiling on recursion depth for tree(), independent of the
# caller's max_depth (which may be None/unlimited). Protects against a
# server bug returning directory entries that don't actually descend.
_HARD_MAX_DEPTH = 100


def resolve_fetch_path(entry: "DMEntry") -> Tuple[str, str]:
    """Derive the `(path, file)` pair `/dm`'s fetch endpoint expects
    from a file `DMEntry`.

    FOXDEN's documented example uses a directory-only `path`
    (`path=/5/meta/1&file=scalars.txt`), but some deployments instead
    return the *full* file path -- including the filename -- as the
    entry's `path` (e.g. `path=1/meta/1/scalars.json` with
    `name=scalars.json`). This strips a trailing `/<name>` off
    `entry.path` when present so both conventions resolve to the same
    correct `(directory, filename)` pair.
    """
    path = entry.path.strip("/")
    name = entry.name
    if path == name:
        dirpath = ""
    elif path.endswith(f"/{name}"):
        dirpath = path[: -(len(name) + 1)]
    else:
        # already directory-only, matching FOXDEN's documented convention
        dirpath = path
    return dirpath, name


@dataclass
class DMEntry:
    """A single file or directory entry under a dataset's DID."""

    did: str
    name: str
    path: str
    is_dir: bool
    esc_did: str = ""

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "DMEntry":
        return cls(
            did=obj.get("did", ""),
            name=obj.get("name", ""),
            path=obj.get("path", ""),
            is_dir=bool(obj.get("is_dir", False)),
            esc_did=obj.get("esc_did", ""),
        )


@dataclass
class DMNode:
    """A node in the recursively-built directory tree for a dataset.

    The synthetic root node (returned by `Dataset.tree()`) has
    `entry is None`; every other node wraps a `DMEntry`.
    """

    entry: Optional[DMEntry]
    children: List["DMNode"] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.entry.name if self.entry else "/"

    @property
    def path(self) -> str:
        return self.entry.path if self.entry else ""

    @property
    def is_dir(self) -> bool:
        return self.entry.is_dir if self.entry else True

    def files(self) -> List["DMNode"]:
        """All file (non-directory) descendants, depth-first."""
        out: List["DMNode"] = []
        for child in self.children:
            if child.is_dir:
                out.extend(child.files())
            else:
                out.append(child)
        return out

    def dirs(self) -> List["DMNode"]:
        """All directory descendants, depth-first (self excluded)."""
        out: List["DMNode"] = []
        for child in self.children:
            if child.is_dir:
                out.append(child)
                out.extend(child.dirs())
        return out

    def walk(self) -> Iterator["DMNode"]:
        """Yield every node in the tree, depth-first (self included)."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, path: str) -> Optional["DMNode"]:
        """Find a descendant node by its path (e.g. '5/meta/1')."""
        target = path.strip("/")
        for node in self.walk():
            if node.path.strip("/") == target:
                return node
        return None

    def pretty(self, indent: int = 0) -> str:
        """Render the (sub)tree as an indented text listing."""
        lines = []
        if self.entry is not None:
            marker = "/" if self.is_dir else ""
            lines.append("  " * indent + f"{self.name}{marker}")
        for child in self.children:
            lines.append(child.pretty(indent + (1 if self.entry is not None else 0)))
        return "\n".join(lines)

    def __repr__(self) -> str:
        kind = "dir" if self.is_dir else "file"
        return f"DMNode({kind} {self.path!r}, {len(self.children)} children)"


class DMClient:
    """Wraps the FOXDEN frontend's `/dm` endpoint."""

    def __init__(self, http: FoxdenClient, base_url: str):
        self._http = http
        self._base_url = base_url

    def list_dir(self, did: str, path: Optional[str] = None) -> List[DMEntry]:
        """List entries directly under `path` (dataset root if `path`
        is None/empty)."""
        params: Dict[str, Any] = {"did": did}
        if path:
            params["path"] = path
        data = self._http.get_raw(self._base_url, "/dm", params=params)
        if not isinstance(data, list):
            # be tolerant in case a future API version wraps results
            data = data.get("results", []) if isinstance(data, dict) else []
        return [DMEntry.from_json(item) for item in data]

    def get_file(self, did: str, path: str, file: str) -> bytes:
        """Fetch a single file's raw bytes."""
        params = {"did": did, "path": path, "file": file}
        return self._http.get_bytes(self._base_url, "/dm", params=params)

    def stream_file(
        self, did: str, path: str, file: str, chunk_size: int = 1 << 20
    ):
        """Yield a file's content in chunks without loading it fully
        into memory."""
        params = {"did": did, "path": path, "file": file}
        yield from self._http.stream(self._base_url, "/dm", params=params, chunk_size=chunk_size)

    def tree(
        self,
        did: str,
        path: Optional[str] = None,
        max_depth: Optional[int] = None,
        progress_every: int = 25,
    ) -> DMNode:
        """Recursively walk the dataset's directory tree starting at
        `path` (dataset root if omitted) and return it as a `DMNode`.

        Guards against a server returning a directory entry that
        doesn't actually descend (which would otherwise recurse
        forever): each visited path is tracked, and a hard safety cap
        (`_HARD_MAX_DEPTH`) applies even when `max_depth` is None.

        Large trees can take a while to walk (one `/dm` request per
        directory) -- logs an INFO progress line every
        `progress_every` directories listed (set to 0 to disable), so
        it's visible that the walk is making progress rather than
        stuck. Enable with `logging.basicConfig(level=logging.INFO)`.
        """
        root = DMNode(entry=None)
        visited: set = set()
        counts = {"dirs": 0, "files": 0}
        self._fill(
            root, did, path, depth=0, max_depth=max_depth, visited=visited,
            counts=counts, progress_every=progress_every,
        )
        if progress_every:
            logger.info(
                "foxden: tree walk complete -- %d directories, %d files",
                counts["dirs"], counts["files"],
            )
        return root

    def _fill(
        self,
        node: DMNode,
        did: str,
        path: Optional[str],
        depth: int,
        max_depth: Optional[int],
        visited: set,
        counts: Dict[str, int],
        progress_every: int,
    ) -> None:
        if max_depth is not None and depth >= max_depth:
            return
        if depth >= _HARD_MAX_DEPTH:
            logger.warning(
                "foxden: /dm tree exceeded safety depth of %d levels at path=%r; "
                "stopping recursion here (pass max_depth= to silence this)",
                _HARD_MAX_DEPTH,
                path,
            )
            return

        key = (path or "").strip("/")
        if key in visited:
            logger.warning(
                "foxden: /dm returned a cycle back to path=%r; skipping to avoid "
                "an infinite loop",
                path,
            )
            return
        visited = visited | {key}

        entries = self.list_dir(did, path)
        counts["dirs"] += 1
        if progress_every and counts["dirs"] % progress_every == 0:
            logger.info(
                "foxden: tree walk in progress -- %d directories listed, "
                "%d files found so far (currently in %r)",
                counts["dirs"], counts["files"], path or "/",
            )

        for entry in entries:
            child = DMNode(entry=entry)
            node.children.append(child)
            if entry.is_dir:
                self._fill(
                    child, did, entry.path, depth + 1, max_depth, visited,
                    counts, progress_every,
                )
            else:
                counts["files"] += 1
