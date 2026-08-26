# foxden (Python client)

A lightweight Python client for the FOXDEN data ecosystem: resolve a
dataset identifier (DID) to its metadata record via the FOXDEN
frontend's `/record` endpoint, then browse and fetch its files via the
`/dm` data management endpoint.

## Install

```bash
pip install .
# or, for config-file (~/.foxden.yaml) support:
pip install ".[yaml]"
```

## Quick start

```python
import foxden

ds = foxden.open("/beamline=3a/btr=shanks-4838-a/cycle=2026-1/sample_name=ti7-19-tomo")

print(ds.metadata)          # dict, from GET /record?did=...

for batch in ds.training_data(batch_size=128):
    for item in batch:
        item["path"]          # directory the file lives in
        item["name"]          # filename
        item["data"]          # raw bytes
```

## Browsing a dataset's file tree

FOXDEN's `/dm` endpoint is a directory listing API:

```
GET /dm?did=<did>                          -> entries at the dataset root
GET /dm?did=<did>&path=<path>              -> entries under <path>
GET /dm?did=<did>&path=<path>&file=<file>  -> a single file's raw bytes
```

`foxden` wraps this with three levels of browsing:

```python
# 1. Shallow, one-level listing (always a fresh call)
for entry in ds.list():                 # dataset root
    print(entry.name, entry.is_dir, entry.path)

for entry in ds.list(path="1/meta"):    # one level under "1/meta"
    print(entry.name, entry.is_dir)

# 2. Full nested tree (recursively walks /dm, cached after first call)
tree = ds.tree()
print(tree.pretty())
# 1/
#   meta/
#     1/
#       scalars.txt
#       positions.txt
#   data.txt
# 2/
#   info.txt

for node in tree.walk():                # every node, depth-first
    print(node.path, node.is_dir)

# 3. Flat list of every file (recursively), optionally scoped to a subpath
all_files = ds.files()                  # every file in the dataset
some_files = ds.files(path="1/meta/1")  # only files under "1/meta/1"
```

`ds.tree()` and `ds.files()` build on the same cached tree, so calling
both doesn't double the number of `/dm` requests. Use `ds.refresh()` or
`ds.tree(refresh=True)` if the underlying data may have changed.

## Fetching a specific file

```python
files = ds.files()
data = ds.fetch(files[0])                       # recommended: pass a DMEntry directly
data = ds.get_file("1/meta/1", "scalars.json")   # or: raw form, GET /dm?...&path=1/meta/1&file=scalars.json
```

matching FOXDEN's documented example of
`/dm?did=$did&path=/5/meta/1&file=scalars.txt`.

> **Note on `path` conventions:** FOXDEN's docs show a file entry's
> `path` as its parent directory (`path=5/meta/1`, filename passed
> separately as `file=`). Some deployments instead return the entry's
> *full* path including the filename itself (e.g.
> `path=1/meta/1/scalars.json`, `name=scalars.json`). `ds.fetch(entry)`
> normalizes both cases automatically (see `foxden.resolve_fetch_path`),
> so prefer it over building the `path=`/`file=` pair yourself from a
> listed entry. `ds.get_file(path, file)` is the raw two-argument form
> for when you already know the exact directory a file lives in.

## Fetching files in chunks

When a (sub-)path contains many files, `training_data()` recursively
collects them and yields batches, downloading each batch's files in
parallel:

```python
for batch in ds.training_data(path="1/meta/1", batch_size=64):
    ...
```

Omit `path` to batch over every file in the dataset. `training_data()`
also accepts:

- `include`/`exclude` -- case-insensitive glob patterns matched against
  filenames, to scope out large or irrelevant files *before* anything
  is downloaded, e.g. skip big raw detector frames and keep only
  scalar/metadata files:

  ```python
  for batch in ds.training_data(batch_size=64, exclude=["*.h5"]):
      ...
  for batch in ds.training_data(batch_size=64, include=["*.mcs", "*.txt"]):
      ...
  ```

  `Dataset.files()` takes the same `include`/`exclude` arguments if you
  just want the filtered list without downloading.
- `loader(entry, data) -> Any` -- transform each file instead of getting
  the default `{"path", "name", "data"}` dict, e.g.:

  ```python
  import io
  import numpy as np

  def load_npy(entry, data):
      return np.load(io.BytesIO(data))

  for batch in ds.training_data(batch_size=64, loader=load_npy):
      arrays = np.stack(batch)
      ...
  ```
- `shuffle=True`, `seed=...` -- shuffle file order once up front
- `max_workers=...` -- parallel downloads per batch (default 8)
- `progress=True` (default) -- log an INFO line after each batch with
  files/MB downloaded so far, so a large run is visibly making progress

For a single large file (e.g. a multi-hundred-MB `.h5` frame) where
loading the whole thing into memory via `.fetch()` isn't desirable, use
`ds.stream(entry)` to iterate over it in chunks instead.

Real scientific datasets can have deep, wide trees with many large
files -- `training_data()` recursively enumerates the whole (sub-)tree
before yielding its first batch, and downloads can legitimately take a
while. See **Troubleshooting** below for how to tell "still working" apart
from "actually stuck".

## Configuration

Resolved in order of precedence (highest first): explicit arguments to
`foxden.open()` -> environment variables -> `~/.foxden.yaml` -> built-in
defaults.

| Env var                | Meaning                                          | Default                 |
|------------------------|---------------------------------------------------|--------------------------|
| `FOXDEN_URL`            | frontend base URL (`/record`, `/dm`)              | `http://localhost:8344`  |
| `FOXDEN_TOKEN`          | bearer token for `Authorization: Bearer <token>`  | *(none)*                 |
| `FOXDEN_VERIFY_SSL`     | `0`/`false` to disable TLS verification           | `true`                   |
| `FOXDEN_TIMEOUT`        | per-request read timeout, seconds                 | `30`                     |
| `FOXDEN_CONNECT_TIMEOUT`| per-request connect timeout, seconds              | `10`                     |
| `FOXDEN_RETRIES`        | retry attempts on 429/500/502/503/504 or timeout  | `3`                      |

If `FOXDEN_TOKEN` isn't set, a token is read from `~/.foxden/token`
(plain text file) as a fallback.

```python
import foxden

ds = foxden.open("some-did", url="https://foxden.example.org", token="eyJ...")
```

## Troubleshooting a slow or "stuck" `training_data()`

**First, check if it's actually stuck or just working through a lot of
data.** Real scientific datasets can have thousands of files, some
individually hundreds of MB (raw detector frames, etc.), spread across
a deep directory tree that has to be enumerated one `/dm` request per
directory before batching can even start. That can legitimately take
minutes with no visible output by default.

- **Turn on progress logging** -- this is the first thing to try. It
  shows both the tree walk and each batch as they happen:

  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

  You should see lines like `tree walk in progress -- 75 directories
  listed, 210 files found so far` and `training_data progress -- 96/1280
  files, 812.4 MB downloaded`. If these keep advancing, it's working --
  just slow. If they stop advancing entirely for a long time, that's a
  genuine hang.

  For raw request-level detail (URLs, status codes), add
  `logging.getLogger("urllib3").setLevel(logging.DEBUG)` as well --
  but the `foxden`-level INFO lines above are usually enough to tell
  "progressing" from "stuck".

- **Scope down what you actually download** -- if you don't need every
  file (e.g. raw `.h5` frames alongside `.mcs`/`.txt` scalars), use
  `include=`/`exclude=` glob patterns on `.files()` or
  `.training_data()` to skip the large/irrelevant ones before anything
  is fetched. This is usually the biggest win when a dataset is simply
  large.

- **Fail faster on a flaky/slow server** -- lower the timeouts and/or
  retries, e.g. `export FOXDEN_TIMEOUT=5 FOXDEN_RETRIES=1`, or:

  ```python
  ds = foxden.open(did, config=foxden.FoxdenConfig(
      url="https://foxden-dev.example.org:8344",
      timeout=5, connect_timeout=3, retries=1,
  ))
  ```

  Every request is bounded by `(connect_timeout, timeout)` and each
  `training_data()` worker thread uses its own connection (not a
  shared session), so one slow/unresponsive file can't hang the whole
  batch forever -- but with the default `retries=3`, a file that
  consistently times out can still take roughly `timeout` x 4 attempts
  (with backoff) before it gives up and moves on.

- **Rule out concurrency** -- pass `max_workers=1` to `training_data()`
  to download sequentially. If that's more reliable than the default
  `max_workers=8` against your dev server, a lower `max_workers` (2-4)
  is a reasonable permanent setting for it.

- **Rule out a runaway tree** -- `ds.tree(max_depth=...)` (or
  `ds.files(path=..., max_depth=...)`) limits recursion depth. The
  client also guards against a server returning a directory that
  doesn't actually descend (a genuine infinite loop) with its own hard
  safety cap, independent of `max_depth`.

## Errors

All exceptions derive from `foxden.FoxdenError`:

- `AuthenticationError` -- 401/403 from a FOXDEN service
- `RecordNotFoundError` -- DID has no metadata record
- `ServiceError` -- FOXDEN's JSON envelope reports a service-level error
  (`error` / `service_code`) on `/record`

## Examples

See `examples/basic_usage.py` for a runnable script covering metadata,
tree browsing, single-file fetch, and chunked batch iteration.
