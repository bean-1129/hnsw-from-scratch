#!/usr/bin/env python3
"""Convert an ann-benchmarks HDF5 dataset into the flat binaries the C++ engine reads.

The C++ side never links against HDF5; this script is the only place that
understands the archive format. It writes:

    base.fbin                 uint32 count | uint32 dim | float32 count*dim
    query.fbin                same layout
    groundtruth.ibin          uint32 count | uint32 k   | int32   count*k
    subset100k.fbin           first N base vectors, for quick experiments
    groundtruth_subset100k.ibin   ground truth recomputed for that subset

Everything is written in the host's native (little endian) byte order.

Usage
-----
    python python/convert_hdf5.py --input data/glove-100-angular.hdf5 --output-dir data
    python python/convert_hdf5.py --download --output-dir data

The vectors are stored *un-normalized*, exactly as they appear in the archive.
Normalization happens once inside the C++ loader (see `loadVectors`), which is
what makes `cosine distance == 1 - dot` valid during search.
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.request

import numpy as np

DEFAULT_URL = "https://ann-benchmarks.com/glove-100-angular.hdf5"
SUBSET_SIZE = 100_000

# The CDN in front of ann-benchmarks.com answers 403 to the default
# `Python-urllib/x.y` User-Agent, so the downloader identifies itself instead.
USER_AGENT = "hnsw-from-scratch/1.0"
CHUNK_BYTES = 1 << 20


def write_fbin(path: str, vectors: np.ndarray) -> None:
    """Write `uint32 count | uint32 dim | float32 data`."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {vectors.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        np.array([vectors.shape[0], vectors.shape[1]], dtype=np.uint32).tofile(handle)
        vectors.tofile(handle)
    print(f"  wrote {path}: {vectors.shape[0]} x {vectors.shape[1]} float32 "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


def write_ibin(path: str, neighbors: np.ndarray) -> None:
    """Write `uint32 count | uint32 k | int32 ids`."""
    neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
    if neighbors.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {neighbors.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        np.array([neighbors.shape[0], neighbors.shape[1]], dtype=np.uint32).tofile(handle)
        neighbors.tofile(handle)
    print(f"  wrote {path}: {neighbors.shape[0]} x {neighbors.shape[1]} int32 "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


def read_fbin(path: str) -> np.ndarray:
    """Read back a .fbin file (used by the other scripts and for verification)."""
    with open(path, "rb") as handle:
        count, dim = np.fromfile(handle, dtype=np.uint32, count=2)
        data = np.fromfile(handle, dtype=np.float32, count=int(count) * int(dim))
    return data.reshape(int(count), int(dim))


def read_ibin(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        count, k = np.fromfile(handle, dtype=np.uint32, count=2)
        data = np.fromfile(handle, dtype=np.int32, count=int(count) * int(k))
    return data.reshape(int(count), int(k))


def normalized(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; zero rows are left untouched."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def exact_neighbors(base: np.ndarray, queries: np.ndarray, k: int,
                    block: int = 1024) -> np.ndarray:
    """Exact cosine top-k by blocked matrix multiplication.

    Both inputs are normalized first, so a maximal inner product is a minimal
    cosine distance. Blocking keeps the similarity matrix out of memory: the
    full 10k x 1.2M product would be ~50 GB.
    """
    base_unit = normalized(base)
    query_unit = normalized(queries)

    result = np.empty((query_unit.shape[0], k), dtype=np.int32)
    started = time.time()
    for start in range(0, query_unit.shape[0], block):
        stop = min(start + block, query_unit.shape[0])
        similarity = query_unit[start:stop] @ base_unit.T          # (b, n)
        top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        # argpartition does not order the k winners; sort them by similarity.
        rows = np.arange(top.shape[0])[:, None]
        order = np.argsort(-similarity[rows, top], axis=1)
        result[start:stop] = top[rows, order].astype(np.int32)
        print(f"\r  ground truth {stop}/{query_unit.shape[0]} "
              f"({time.time() - started:.1f}s)", end="", flush=True)
    print()
    return result


def _ssl_context() -> "ssl.SSLContext":
    """A verifying SSL context that also works on a bare python.org install.

    The macOS python.org installers ship without a CA bundle until the bundled
    `Install Certificates.command` is run, so HTTPS fails with
    CERTIFICATE_VERIFY_FAILED. Rather than disabling verification (never do
    that for a 460 MB download), fall back to certifi or to the system trust
    store that macOS already maintains at /etc/ssl/cert.pem.
    """
    context = ssl.create_default_context()
    if context.cert_store_stats()["x509_ca"] > 0:
        return context

    try:
        import certifi

        context.load_verify_locations(cafile=certifi.where())
        print("  note: using certifi's CA bundle")
        return context
    except ImportError:
        pass

    system_bundle = "/etc/ssl/cert.pem"
    if os.path.exists(system_bundle):
        context.load_verify_locations(cafile=system_bundle)
        print(f"  note: this Python has no CA bundle; using {system_bundle}")
        return context

    raise RuntimeError(
        "no CA certificates available for HTTPS. On macOS run\n"
        "  '/Applications/Python 3.x/Install Certificates.command'\n"
        "or install certifi with `pip install certifi`.")


def download(url: str, destination: str) -> None:
    """Stream `url` to `destination`, showing progress.

    The transfer goes to a `.part` file that is renamed only once the expected
    number of bytes has arrived. Without that, an interrupted download would
    leave a truncated archive behind, and the `already exists` check below would
    happily skip re-fetching it on the next run.
    """
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping download")
        return

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    partial = destination + ".part"
    print(f"Downloading {url} -> {destination}")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = _ssl_context() if url.lower().startswith("https") else None
    started = time.time()
    try:
        with urllib.request.urlopen(request, context=context) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    elapsed = max(time.time() - started, 1e-6)
                    if total > 0:
                        print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB "
                              f"({100.0 * done / total:5.1f}%, "
                              f"{done / 1e6 / elapsed:.1f} MB/s)", end="", flush=True)
                    else:
                        print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
        print()
        if total > 0 and done != total:
            raise IOError(f"expected {total} bytes, received {done}")
    except BaseException:
        # Includes KeyboardInterrupt: never leave a half-written archive around.
        if os.path.exists(partial):
            os.remove(partial)
        raise

    os.replace(partial, destination)
    print(f"  saved {os.path.getsize(destination) / 1e6:.1f} MB to {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an ann-benchmarks HDF5 dataset to .fbin/.ibin files")
    parser.add_argument("--input", default="data/glove-100-angular.hdf5",
                        help="path to the HDF5 archive")
    parser.add_argument("--output-dir", default="data",
                        help="directory that receives the binary files")
    parser.add_argument("--download", action="store_true",
                        help="fetch the archive from ann-benchmarks.com first")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="download URL used with --download")
    parser.add_argument("--subset", type=int, default=SUBSET_SIZE,
                        help="size of the quick-experiment subset (0 disables it)")
    parser.add_argument("--subset-groundtruth", action="store_true", default=True,
                        help="recompute exact neighbors for the subset (default: on)")
    parser.add_argument("--no-subset-groundtruth", dest="subset_groundtruth",
                        action="store_false")
    parser.add_argument("--k", type=int, default=100,
                        help="neighbors per query for the recomputed subset ground truth")
    args = parser.parse_args()

    if args.download:
        download(args.url, args.input)

    try:
        import h5py  # imported late so --download works without h5py installed
    except ImportError:
        print("error: h5py is required. Install it with `pip install h5py numpy`.",
              file=sys.stderr)
        return 1

    if not os.path.exists(args.input):
        print(f"error: {args.input} not found. Pass --download to fetch it.",
              file=sys.stderr)
        return 1

    print(f"Reading {args.input}")
    with h5py.File(args.input, "r") as archive:
        missing = [key for key in ("train", "test", "neighbors") if key not in archive]
        if missing:
            print(f"error: {args.input} is missing dataset(s): {missing}", file=sys.stderr)
            return 1
        base = np.array(archive["train"], dtype=np.float32)
        queries = np.array(archive["test"], dtype=np.float32)
        groundtruth = np.array(archive["neighbors"], dtype=np.int32)
        distance_metric = archive.attrs.get("distance", b"unknown")

    if isinstance(distance_metric, bytes):
        distance_metric = distance_metric.decode()
    print(f"  base   : {base.shape}   queries: {queries.shape}")
    print(f"  truth  : {groundtruth.shape}   metric: {distance_metric}")

    out = args.output_dir
    write_fbin(os.path.join(out, "base.fbin"), base)
    write_fbin(os.path.join(out, "query.fbin"), queries)
    write_ibin(os.path.join(out, "groundtruth.ibin"), groundtruth)

    if args.subset > 0:
        size = min(args.subset, base.shape[0])
        subset = base[:size]
        name = f"subset{size // 1000}k"
        write_fbin(os.path.join(out, f"{name}.fbin"), subset)

        if args.subset_groundtruth:
            # The shipped ground truth indexes the *full* base set, so it is
            # wrong for a subset. Recompute it, otherwise recall measured on the
            # subset is meaningless.
            print(f"Recomputing exact neighbors for {name} (k={args.k})")
            neighbors = exact_neighbors(subset, queries, args.k)
            write_ibin(os.path.join(out, f"groundtruth_{name}.ibin"), neighbors)

    # Cheap self-check: the first query's shipped ground truth must agree with a
    # direct computation on the full base set.
    probe = normalized(queries[:1]) @ normalized(base).T
    expected = int(np.argmax(probe))
    if expected != int(groundtruth[0, 0]):
        print(f"warning: shipped ground truth says {groundtruth[0, 0]} for query 0, "
              f"exact cosine search says {expected}")
    else:
        print("Verified: shipped ground truth matches an exact cosine search for query 0")

    print("\nDone. Build an index with:")
    print(f"  ./build/bin/hnsw build --base {os.path.join(out, 'base.fbin')} "
          f"--out results/index.hnsw --M 16 --ef-construction 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
