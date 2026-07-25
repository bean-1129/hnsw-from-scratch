#!/usr/bin/env python3
"""FAISS reference run, for an apples-to-apples comparison with the C++ engine.

FAISS is used *only* here, as an external yardstick. The C++ implementation in
`src/` does not link against FAISS (or any other ANN library) at all.

The measurement protocol mirrors `src/benchmark.cpp` exactly:

  * the same base/query/ground-truth binaries,
  * batch size 1 (one query per search call),
  * a single thread (`faiss.omp_set_num_threads(1)`),
  * the same warm-up, the same efSearch sweep,
  * the same CSV columns: ef,recall,median_ms,p95_ms,qps

Install with:  pip install faiss-cpu numpy

Usage:
    python python/faiss_reference.py \
        --base data/base.fbin --query data/query.fbin --gt data/groundtruth.ibin \
        --M 16 --ef-construction 200 --csv results/faiss.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def read_fbin(path: str, limit: int = 0) -> np.ndarray:
    with open(path, "rb") as handle:
        count, dim = np.fromfile(handle, dtype=np.uint32, count=2)
        count, dim = int(count), int(dim)
        if limit > 0:
            count = min(count, limit)
        data = np.fromfile(handle, dtype=np.float32, count=count * dim)
    return data.reshape(count, dim)


def read_ibin(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        count, k = np.fromfile(handle, dtype=np.uint32, count=2)
        data = np.fromfile(handle, dtype=np.int32, count=int(count) * int(k))
    return data.reshape(int(count), int(k))


def normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def recall_at_k(predicted: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean |predicted_k ∩ truth_k| / k -- identical to `recall.cpp`."""
    hits = 0
    for row in range(predicted.shape[0]):
        hits += len(set(predicted[row, :k].tolist()) & set(truth[row, :k].tolist()))
    return hits / (predicted.shape[0] * k)


def main() -> int:
    parser = argparse.ArgumentParser(description="FAISS IndexHNSWFlat reference benchmark")
    parser.add_argument("--base", default="data/base.fbin")
    parser.add_argument("--query", default="data/query.fbin")
    parser.add_argument("--gt", default="data/groundtruth.ibin")
    parser.add_argument("--csv", default="results/faiss.csv")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-list", default="16,32,64,128,256,512")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N base vectors")
    parser.add_argument("--queries", type=int, default=0,
                        help="use only the first N queries")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1,
                        help="OpenMP threads; keep at 1 to match the C++ numbers")
    args = parser.parse_args()

    try:
        import faiss
    except ImportError:
        print("error: faiss is not installed. Run `pip install faiss-cpu`.",
              file=sys.stderr)
        return 1

    for path in (args.base, args.query, args.gt):
        if not os.path.exists(path):
            print(f"error: {path} not found (run python/convert_hdf5.py first)",
                  file=sys.stderr)
            return 1

    # Normalize once, exactly like the C++ loader does.
    base = normalized(read_fbin(args.base, args.limit))
    queries = normalized(read_fbin(args.query))
    truth = read_ibin(args.gt)
    if args.queries > 0:
        queries = queries[:args.queries]
        truth = truth[:args.queries]

    print(f"base {base.shape}, queries {queries.shape}, ground truth {truth.shape}")
    faiss.omp_set_num_threads(args.threads)

    # For unit-length vectors the L2 and cosine orderings coincide
    # (||a-b||^2 = 2 - 2*dot), so METRIC_L2 gives cosine ranking here.
    index = faiss.IndexHNSWFlat(base.shape[1], args.M)
    index.hnsw.efConstruction = args.ef_construction

    print(f"Building IndexHNSWFlat(M={args.M}, efConstruction={args.ef_construction})")
    started = time.perf_counter()
    index.add(base)
    build_seconds = time.perf_counter() - started
    print(f"  built in {build_seconds:.1f} s "
          f"({base.shape[0] / build_seconds:.0f} vectors/s)")

    ef_values = [int(value) for value in args.ef_list.split(",") if value]
    rows = []
    for ef in ef_values:
        index.hnsw.efSearch = ef

        warmup = min(args.warmup, queries.shape[0])
        for i in range(warmup):
            index.search(queries[i:i + 1], args.k)

        latencies = np.empty(queries.shape[0], dtype=np.float64)
        predicted = np.empty((queries.shape[0], args.k), dtype=np.int64)

        wall = time.perf_counter()
        for i in range(queries.shape[0]):
            single = time.perf_counter()
            _, ids = index.search(queries[i:i + 1], args.k)
            latencies[i] = (time.perf_counter() - single) * 1000.0
            predicted[i] = ids[0]
        seconds = time.perf_counter() - wall

        recall = recall_at_k(predicted, truth, args.k)
        row = {
            "ef": ef,
            "recall": recall,
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "qps": queries.shape[0] / seconds,
        }
        rows.append(row)
        print(f"  ef={ef:<5} recall@{args.k}={recall:.4f}  "
              f"median={row['median_ms']:.3f} ms  p95={row['p95_ms']:.3f} ms  "
              f"{row['qps']:.0f} qps")

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w") as handle:
        handle.write("ef,recall,median_ms,p95_ms,qps\n")
        for row in rows:
            handle.write(f"{row['ef']},{row['recall']:.6f},{row['median_ms']:.6f},"
                         f"{row['p95_ms']:.6f},{row['qps']:.2f}\n")
    print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
