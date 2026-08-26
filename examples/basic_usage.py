"""Basic usage example for the foxden package.

Set FOXDEN_URL / FOXDEN_TOKEN (or edit the values below) before running:

    export FOXDEN_URL=http://localhost:8344
    export FOXDEN_TOKEN=$(cat ~/.foxden/token)
    python examples/basic_usage.py "/beamline=3a/btr=shanks-4838-a/cycle=2026-1/sample_name=ti7-19-tomo"

Real datasets can have large, deep trees with big binary files (raw
detector frames, etc.), so this example:
  - enables INFO-level progress logging by default (tree walk + batch
    downloads), so a long run is visibly making progress rather than
    looking stuck -- see the "Troubleshooting" section of the README;
  - excludes a couple of large/raw file extensions by default in the
    training_data() step, so running this script doesn't accidentally
    pull down gigabytes of data. Adjust EXCLUDE_PATTERNS below (or pass
    include=[...] instead) to match your own dataset.
"""
import logging
import sys

import foxden
import logging
logging.basicConfig(level=logging.INFO)

# Skip these by default in the training_data() demo below -- edit to
# match what your dataset actually contains.
EXCLUDE_PATTERNS = ["*.h5", "*.hdf5"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    did = sys.argv[1] if len(sys.argv) > 1 else "CHESS-ABC123"

    ds = foxden.open(did)

    # 1. Metadata
    print("=== metadata ===")
    print(ds.metadata)

    # 2. Explore the dataset's nested file tree (logs tree-walk progress)
    print("\n=== tree ===")
    print(ds.tree().pretty())

    # 3. Shallow listing of a specific subdirectory
    print("\n=== list(path='1') ===")
    for entry in ds.list(path="1"):
        kind = "dir" if entry.is_dir else "file"
        print(f"  [{kind}] {entry.path}/{entry.name}")

    # 4. Fetch one specific (small) file directly
    small_files = ds.files(exclude=EXCLUDE_PATTERNS)
    if small_files:
        first = small_files[0]
        print(f"\n=== fetch({first.path!r}/{first.name!r}) ===")
        data = ds.fetch(first)
        print(f"  {len(data)} bytes")

    # 5. Iterate the dataset in chunks (logs per-batch progress)
    print(f"\n=== training_data(batch_size=32, exclude={EXCLUDE_PATTERNS}) ===")
    total = 0
    for i, batch in enumerate(ds.training_data(batch_size=32, exclude=EXCLUDE_PATTERNS)):
        total += len(batch)
        print(f"  batch {i}: {len(batch)} files")
    print(f"  total files processed: {total}")


if __name__ == "__main__":
    main()
