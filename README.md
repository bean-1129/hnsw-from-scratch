# hnsw-from-scratch

A complete, dependency-free C++17 implementation of **Hierarchical Navigable Small World** graphs for approximate nearest neighbor search, following

> Yu. A. Malkov, D. A. Yashunin.
> *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.*
> IEEE TPAMI 2018 ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320))

Every algorithm in the paper — the insertion procedure, the layer search, both neighbor selection strategies and the k-NN search — is implemented here from the pseudocode. **No hnswlib, no FAISS, no ANN library of any kind is linked into the C++ engine.** FAISS appears exactly once, in a standalone Python script, as an external yardstick to measure this implementation against.

---

## Table of contents

- [What is in the box](#what-is-in-the-box)
- [Architecture](#architecture)
- [The algorithms](#the-algorithms)
- [Parameters](#parameters)
- [Data format](#data-format)
- [Building](#building)
- [Usage](#usage)
- [Dataset preparation](#dataset-preparation)
- [Benchmarking](#benchmarking)
- [Measured results](#measured-results)
- [Performance discussion](#performance-discussion)
- [Testing](#testing)
- [Engineering maturity](#engineering-maturity)
- [References](#references)

---

## What is in the box

| Component | Where | Notes |
|---|---|---|
| Binary vector loader | `src/vector_loader.cpp` | `.fbin` / `.ibin`, contiguous storage, normalize-once |
| Distance kernels | `src/distance.cpp` | dot product, cosine, squared L2 — plain scalar loops |
| Exact brute-force search | `src/brute_force.cpp` | correctness oracle, multi-threaded batch mode |
| Recall@k evaluator | `src/recall.cpp` | the ann-benchmarks definition |
| Layered graph | `src/graph.cpp` | topology, degree caps, serialization, statistics |
| HNSW index | `src/hnsw.cpp` | Algorithms 1–5 of the paper |
| Benchmark harness | `src/benchmark.cpp` | efSearch sweep, latency percentiles, CSV output |
| CLI | `src/main.cpp` | `build`, `search`, `benchmark`, `save`, `load`, `bruteforce` |
| Dataset conversion | `python/convert_hdf5.py` | HDF5 → `.fbin` / `.ibin` |
| FAISS comparison | `python/faiss_reference.py` | identical protocol, identical CSV schema |
| Unit tests | `tests/` | 5 binaries, no test framework needed |

## Architecture

```
                    python/convert_hdf5.py
  glove-100-angular.hdf5  ──────────────►  base.fbin  query.fbin  groundtruth.ibin
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │  vector_loader.hpp     │  normalize once
                                   │  VectorSet (contiguous)│  ⇒ cosine = 1 - dot
                                   └───────────┬────────────┘
                                               │
                      ┌────────────────────────┼────────────────────────┐
                      ▼                        ▼                        ▼
            ┌──────────────────┐    ┌────────────────────┐    ┌──────────────────┐
            │ brute_force.hpp  │    │     hnsw.hpp       │    │   distance.hpp   │
            │ exact top-k      │    │ Algorithms 1..5    │◄───│ dot / cosine / L2│
            │ (oracle)         │    │  ┌──────────────┐  │    └──────────────────┘
            └────────┬─────────┘    │  │  graph.hpp   │  │
                     │              │  │  topology    │  │
                     │              │  └──────────────┘  │
                     │              └─────────┬──────────┘
                     │                        │
                     └───────────┬────────────┘
                                 ▼
                     ┌───────────────────────┐        ┌──────────────────┐
                     │    recall.hpp         │◄───────│  benchmark.hpp   │──► results/*.csv
                     │    Recall@k           │        │  ef sweep, p50/p95│
                     └───────────────────────┘        └──────────────────┘
```

Two design decisions shape the whole codebase:

**1. Topology and geometry are separated.** `Graph` owns only the adjacency lists (per node, per layer). `HnswIndex` owns the contiguous vector store and every algorithm. The result is that `hnsw.cpp` reads like the paper's pseudocode, and the graph can be inspected, serialized and unit-tested on its own.

**2. Vectors are normalized exactly once, at load time.** After `loadVectors(..., Metric::Cosine)`, every vector has unit length, so

```
cosine_similarity(a, b) = dot(a, b)          cosine_distance(a, b) = 1 - dot(a, b)
```

No norm, no square root and no division is ever computed inside a search loop. The CLI warns if it is handed data that is not unit length, since that is the single most likely cause of unexplained recall loss.

## The algorithms

The mapping from the paper to the code is one-to-one:

| Paper | Code | Purpose |
|---|---|---|
| Algorithm 1 — INSERT | `HnswIndex::addPoint` | full insertion procedure |
| Algorithm 2 — SEARCH-LAYER | `HnswIndex::searchLayer` | beam search inside one layer |
| Algorithm 3 — SELECT-NEIGHBORS-SIMPLE | `HnswIndex::selectNeighborsSimple` | naive "keep the M closest" |
| Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC | `HnswIndex::selectNeighborsHeuristic` | diversity pruning (the default) |
| Algorithm 5 — K-NN-SEARCH | `HnswIndex::searchKnn` | greedy descent + layer-0 beam search |

### Insertion (Algorithm 1)

1. Draw a level from the exponentially decaying distribution
   `l = floor(-ln(U(0,1)) · mL)` with `mL = 1 / ln(M)` (section 4.1).
2. Greedily descend from the top layer down to `l + 1`, using `SEARCH-LAYER` with `ef = 1`.
3. From `min(L, l)` down to layer 0: run `SEARCH-LAYER` with `efConstruction`, pick neighbors with the heuristic, and insert **bidirectional** edges.
4. Whenever a reverse edge overflows the degree cap (`Mmax` above layer 0, `Mmax0 = 2M` on layer 0), re-run the heuristic **from that neighbor's own point of view** to decide which link to drop.
5. If the new element's level exceeds the current top level, it becomes the new entry point.

Following the pseudocode literally, the entire candidate set `W` — not just its nearest element — is carried down as the entry point set for the next layer.

### Neighbor selection (Algorithm 4)

The heuristic is what separates a real HNSW from a k-NN graph. A candidate `e` is kept only if it is closer to the base element than to **any already selected neighbor**:

```
keep e  ⟺  dist(e, base) < min over r in R of dist(e, r)
```

This is a relative-neighborhood-graph style rule. It deliberately refuses links that point into a region already covered by an existing link, which preserves the long-range edges that make the graph navigable, and prevents a densely clustered region from consuming every link slot of a node. Both optional flags of the paper are implemented:

- `extendCandidates` — also consider the neighbors of the candidates (off by default; useful for strongly clustered data).
- `keepPrunedConnections` — top up the selection with the best rejected candidates so the node reaches its full out-degree (on by default).

### Search (Algorithm 5)

Layers `L … 1` are traversed greedily (`ef = 1`); layer 0 runs a beam search with the configurable `efSearch`. `ef` is silently raised to `k` when a caller asks for more neighbors than the beam width.

`SEARCH-LAYER` uses two heaps over one shared scratch buffer (a min-heap of candidates to expand, a max-heap of the best `ef` found) plus an **epoch-tagged visited list**: a `uint32` generation counter per node makes "clear the visited set" an O(1) operation instead of O(n), and removes the per-query allocation that otherwise dominates a short query.

The scratch buffer lives in a `SearchContext`. `searchKnn` uses a `thread_local` one by default, or you can pass your own — several threads may query one `const HnswIndex` concurrently with no shared mutable state.

## Parameters

| Parameter | Flag | Default | Effect |
|---|---|---|---|
| `M` | `--M` | 16 | links per element per layer; `Mmax0 = 2M` on layer 0 |
| `efConstruction` | `--ef-construction` | 200 | candidate list size while building; higher = better graph, slower build |
| `efSearch` | `--ef` | 64 | candidate list size at query time; the recall/latency dial |
| seed | `--seed` | 100 | RNG seed for level assignment; fixing it makes builds bit-reproducible |
| metric | `--metric` | `cosine` | `cosine` (normalized) or `l2` |
| `mL` | — | `1 / ln(M)` | level multiplier, the paper's optimum |

`M` and `efConstruction` are build-time decisions; `efSearch` can be changed per query without rebuilding, which is why the benchmark sweeps it.

Practical guidance, backed by the [parameter study](#parameter-study-100k-subset-same-10000-queries) below: **spend your build budget on `M` before `efConstruction`.** The default `M = 16` is a reasonable starting point up to ~10⁵ vectors, but for a million-scale, high-dimensional dataset such as GloVe, `M = 32` is worth roughly ten times more recall than raising `efConstruction` from 200 to 500 — at the cost of a proportionally larger graph and a slower build.

## Data format

Two flat, header-prefixed binary formats in native (little endian) byte order. **The C++ code never reads HDF5.**

`*.fbin` — vectors:

```
offset 0   uint32   number_of_vectors
offset 4   uint32   dimension
offset 8   float32  vectors[number_of_vectors * dimension]   (row major)
```

`*.ibin` — ground truth neighbor ids:

```
offset 0   uint32   number_of_queries
offset 4   uint32   k
offset 8   int32    neighbors[number_of_queries * k]         (row major)
```

The serialized index (`*.hnsw`) is a third format: a magic header, the parameter block, the raw vector store, then the per-node levels and adjacency lists. It is written by `HnswIndex::save` and read by `HnswIndex::load`.

## Building

Requires CMake ≥ 3.14 and a C++17 compiler. There are no third-party dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The binary lands at `build/bin/hnsw`, the static library at `build/lib/libhnsw_core.a`.

CMake options:

| Option | Default | Meaning |
|---|---|---|
| `HNSW_BUILD_TESTS` | `ON` | build the five test binaries |
| `HNSW_ENABLE_NATIVE_ARCH` | `OFF` | add `-march=native` (or `-mcpu=native`); faster, non-portable binary |
| `HNSW_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` |

Release builds compile with `-O3`; all targets are compiled with `-Wall -Wextra -Wpedantic` (`/W4` on MSVC).

## Usage

```
hnsw <command> [options]

  build       Build an index from a .fbin base file (optionally save it)
  search      Run k-NN queries and report recall / latency
  benchmark   Sweep efSearch and write ef,recall,median_ms,p95_ms,qps as CSV
  save        Build an index and persist it (build with a mandatory --out)
  load        Load an index from disk and print its statistics
  bruteforce  Exact search; writes a ground truth .ibin with --out
```

Build and persist an index:

```bash
./build/bin/hnsw build \
    --base data/base.fbin --out results/index.hnsw \
    --M 16 --ef-construction 200 --seed 42
```

The build prints live progress and then a structural report (a real run over the full GloVe dataset — see [Measured results](#measured-results)):

```
  inserted 1183514 / 1183514 (100.0%)  2162 vec/s  547.5 s elapsed
Built index in 547.49 s (2162 vectors/s)
Index: 1183514 elements, dim 100, metric cosine
  M=16  efConstruction=200  seed=42  entry point=11646  top layer=5
  edges: 30600519   memory: vectors 451.47 MiB + graph 235.74 MiB
  layer 0 :   1183514 nodes, average degree  24.79
  layer 1 :     73770 nodes, average degree  16.00
  layer 2 :      4627 nodes, average degree  16.00
  layer 3 :       290 nodes, average degree  16.00
  layer 4 :        21 nodes, average degree  16.00
  layer 5 :         1 nodes, average degree   0.00
```

Load it back and check recall:

```bash
./build/bin/hnsw load --index results/index.hnsw \
    --query data/query.fbin --gt data/groundtruth.ibin --verify --k 10 --ef 64
```

Query and inspect individual results:

```bash
./build/bin/hnsw search --index results/index.hnsw \
    --query data/query.fbin --gt data/groundtruth.ibin --k 10 --ef 100 --show 5
```

Regenerate ground truth for a subset (exact, multi-threaded):

```bash
./build/bin/hnsw bruteforce --base data/subset100k.fbin \
    --query data/query.fbin --k 100 --out data/groundtruth_subset100k.ibin
```

Every command accepts `--limit N` (use the first N base vectors) and `--queries N`, which makes it easy to iterate on a small slice before running the full dataset. Unknown flags are rejected rather than silently ignored.

## Dataset preparation

The reference dataset is **glove-100-angular** (1,183,514 base vectors, 100 dimensions, 10,000 queries, cosine/angular distance) from [ann-benchmarks.com](https://ann-benchmarks.com).

```bash
pip install numpy h5py
python python/convert_hdf5.py --download --output-dir data
```

This writes `data/base.fbin`, `data/query.fbin`, `data/groundtruth.ibin`, plus `data/subset100k.fbin` and a **recomputed** `data/groundtruth_subset100k.ibin`. The recomputation matters: the shipped ground truth indexes the full base set, so it is simply wrong for a subset — measuring recall on `subset100k.fbin` against `groundtruth.ibin` would produce meaningless numbers.

See [`data/README.md`](data/README.md) for details.

## Benchmarking

```bash
./build/bin/hnsw benchmark \
    --index results/index.hnsw \
    --query data/query.fbin \
    --gt data/groundtruth.ibin \
    --k 10 \
    --ef-list 16,32,64,128,256,512 \
    --csv results/hnsw_cpp.csv
```

Protocol: one query per call (batch size 1), single-threaded, 100 warm-up queries per `ef` value that are excluded from the statistics. Output CSV:

```
ef,recall,median_ms,p95_ms,qps
```

Add `--brute-force --base data/base.fbin` to also time the exact scan and print the speedup at every `ef`.

To compare against FAISS with an identical protocol:

```bash
pip install faiss-cpu
python python/faiss_reference.py \
    --base data/base.fbin --query data/query.fbin --gt data/groundtruth.ibin \
    --M 16 --ef-construction 200 --csv results/faiss.csv
```

Both scripts emit the same five columns, so the two CSVs can be diffed or plotted directly against each other.

## Measured results

All numbers below are real runs on **glove-100-angular** (1,183,514 × 100, all 10,000 queries, `k = 10`), on an **Apple M4, single thread, Release build (`-O3`, no `-march=native`)**.

### Full dataset, M = 16, efConstruction = 200

| ef | Recall@10 | median (ms) | p95 (ms) | QPS | speedup vs. brute force |
|---|---|---|---|---|---|
| 16 | 0.5702 | 0.053 | 0.088 | 17,529 | 447.6× |
| 32 | 0.6789 | 0.088 | 0.138 | 10,734 | 270.9× |
| 64 | 0.7666 | 0.155 | 0.225 | 6,278 | 154.3× |
| 128 | 0.8344 | 0.281 | 0.376 | 3,555 | 85.1× |
| 256 | 0.8866 | 0.516 | 0.650 | 1,968 | 46.3× |
| 512 | 0.9274 | 0.955 | 1.198 | 1,069 | 25.0× |

Exact brute force over the same base set: **23.892 ms** median, 41.8 QPS (measured on 1,000 queries), recall 1.0 by definition.

Build: 1,183,514 vectors in **547 s** (2,162 vectors/s, single-threaded). Memory: 451 MiB of vectors + 236 MiB of graph, 30,600,519 directed edges. Level occupancy decayed as 1,183,514 → 73,770 → 4,627 → 290 → 21 → 1 — a factor of 1/16.04 at the first step, against the 1/M = 1/16 the theory predicts for `mL = 1/ln(M)`.

### Parameter study (100k subset, same 10,000 queries)

`M = 16` is the *low* end of the useful range for GloVe, and the table above shows it: recall tops out in the low 0.9s. The subset study isolates why.

| M | efConstruction | Recall@10 (ef=64) | Recall@10 (ef=512) |
|---|---|---|---|
| 16 | 200 | 0.8350 | 0.9790 |
| 16 | 500 | 0.8353 | 0.9819 |
| 32 | 200 | 0.9043 | 0.9968 |
| 32 | 500 | 0.9056 | **0.9976** |

**`M` sets the recall ceiling; `efConstruction` is a second-order refinement.** Going from `M=16` to `M=32` buys +0.07 recall at ef=64 and closes almost the whole remaining gap at ef=512, while raising `efConstruction` from 200 to 500 is worth only ~0.003. This is exactly the behaviour the paper describes: `M` controls how well the graph approximates the true proximity structure, and no amount of query-time beam width fully compensates for a graph that is too sparse. It is also why ann-benchmarks runs its best hnswlib curves on this dataset at `M` of 32–48, not 16.

The recall ceiling additionally tightens as `n` grows at fixed `M`: `M=16` reaches 0.979 at ef=512 on 100k vectors but only 0.927 on the full 1.18M. Budget `M` by dataset size, not by intuition.

### Reading the speedup column

An exact scan is `O(n·d)` per query; an HNSW search is roughly `O(ef · M · d · log n)`. The advantage therefore grows with `n` — at 1.18M vectors, HNSW is **25× faster than brute force even at its most conservative setting** (ef=512, 93% recall) and 154× faster at 77% recall. The exact scan needs 23.9 ms per query; the graph answers in under 1 ms.

## Performance discussion

### Against brute force

Brute force is exact and cache-friendly — it streams contiguous memory and vectorizes perfectly — but it is linear in the dataset size. HNSW is logarithmic-ish but pays with **random access**: every hop follows a pointer to an unpredictable location in the vector store. That is why the crossover point matters. Below a few thousand vectors, a linear scan wins outright; above roughly 10⁴ it loses badly, and the gap widens with `n`.

### Against FAISS

`faiss::IndexHNSWFlat` implements the same algorithm. Expect it to be meaningfully faster per query at equal recall — the gap is *not* algorithmic, it is engineering, and it comes from four places:

**1. SIMD.** The kernels in `distance.cpp` are deliberately plain scalar loops, as specified. At `-O3` the compiler auto-vectorizes them, but not as well as hand-written intrinsics: FAISS dispatches to AVX2/AVX-512 (or NEON) kernels with explicit unrolling, FMA and horizontal reductions tuned per dimension. For 100-dimensional float32, a hand-tuned kernel is typically **2–4×** faster than the auto-vectorized loop. Enabling `-DHNSW_ENABLE_NATIVE_ARCH=ON` recovers part of this at zero code cost; the next step would be `distance_avx2.cpp` / `distance_neon.cpp` selected at runtime.

**2. Cache locality and prefetching.** During a beam search the CPU stalls on the vector fetch for each neighbor. FAISS (and hnswlib) issue `_mm_prefetch` on the next neighbor's vector and adjacency list while computing the current distance, hiding most of the latency. This implementation does none of that; adding software prefetch into the neighbor loop of `searchLayer` is the single highest-value optimization left, typically worth 20–40%.

**3. Memory layout.** Here, adjacency lists are `std::vector<std::vector<uint32_t>>` — one heap allocation per node per layer, so neighbors are scattered and each hop costs an extra indirection. Production implementations store layer 0 as a **single flat array** with a fixed stride (`Mmax0 + 1` slots per node) and, crucially, **interleave the vector with its neighbor list** so that one cache line fetch delivers both. That layout alone is worth a large fraction of the remaining gap. The interface here is designed so this change stays inside `graph.cpp`.

**4. Quantization.** FAISS can put a scalar-quantized or PQ-compressed vector store behind the same graph (`IndexHNSWSQ`, `IndexHNSWPQ`), cutting memory traffic 4× at a small recall cost. This project stores raw `float32` only.

Two further gaps are about scale rather than speed per query:

- **Parallel construction.** Building is single-threaded here (correctness first: no locks, fully deterministic given a seed). FAISS and hnswlib insert concurrently with per-node locks, which is a near-linear speedup on the build. The search path is already thread-safe.
- **Batch queries.** The benchmark measures batch size 1 to model online serving. Batched search amortizes memory latency across queries and looks better on a throughput chart.

What this implementation does *not* concede is recall: at equal `M`, `efConstruction` and `efSearch`, the graph built here should reach recall parity with FAISS, because the construction algorithm — including the diversity heuristic and the reverse-edge repair — follows the paper exactly. If the recall column of the two CSVs matches while the latency column does not, the difference is entirely in the four items above.

### Engineering trade-offs taken deliberately

| Choice | Cost | Why |
|---|---|---|
| Scalar distance kernels | 2–4× on distance computation | required by the project spec; keeps the code readable and portable |
| `vector<vector<uint32_t>>` adjacency | extra indirection per hop | keeps `Graph` simple and independently testable |
| Single-threaded build | slower ingest | deterministic, reproducible, no lock reasoning |
| `float32` vector store | 4× memory vs. int8 | exactness of the reported distances |
| Epoch-tagged visited list | one `uint32` per node | removes the per-query allocation that dominates short queries |

## Testing

```bash
ctest --test-dir build --output-on-failure
```

Five self-contained binaries, no external test framework:

| Test | Covers |
|---|---|
| `test_distance` | dot product, cosine, squared L2, normalization, zero vectors, metric dispatch, and the identity `1 - dot == ½‖a−b‖²` on unit vectors |
| `test_loader` | `.fbin`/`.ibin` round trips, the exact on-disk header layout, partial loads, normalize-on-load, missing and truncated files |
| `test_bruteforce` | known geometric layouts, sorted/unique results, agreement with a full sort, `k > n`, threaded batch mode, L2 metric |
| `test_hnsw` | see below |
| `test_recall` | perfect/zero/partial recall, order insensitivity, k smaller than the ground truth width, short result rows, invalid arguments |

`test_hnsw` is the important one. It asserts **structural invariants** (degree caps respected per layer, no self loops, no duplicate edges, every neighbor exists on the layer it is linked on, no isolated nodes, geometric level decay), **search correctness** (every point finds itself; recall against the brute-force oracle on both isotropic and clustered data; monotonicity of recall in `ef`; results sorted with distances that match a direct recomputation), **determinism** (two builds with the same seed are identical; a different seed is not), **serialization** (a round trip reproduces the graph edge for edge and the query results exactly; garbage input is rejected), and the behaviour of both Algorithm 4 flags.

## Engineering maturity

**What is production-shaped.** Header/source separation with a documented public API; RAII throughout with no owning raw pointers and no manual `new`/`delete`; const-correct interfaces (searching takes a `const` index); errors reported as exceptions with actionable messages rather than silent failure or `assert`; deterministic, seeded construction; a versioned, magic-checked serialization format that refuses to load a foreign or truncated file; a CLI that validates its own flags; a benchmark harness that reports percentiles rather than only averages.

**What it is not, yet.** This is a portfolio-grade single-node engine, not a vector database. It has no deletions or updates (HNSW supports soft deletes with tombstones and periodic rebuilds — not implemented), no concurrent insertion, no memory-mapped index (`load` reads the whole file into RAM, so the index must fit in memory), no filtered or hybrid search, no quantization, and no persistence beyond a single flat file. The distance kernels are scalar by design.

**Roadmap, in the order that would actually pay off:** software prefetch in `searchLayer` → flat, stride-based layer-0 storage with the vector interleaved → SIMD kernels with runtime dispatch → parallel construction → memory-mapped load → soft deletes.

## References

1. Yu. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 2018. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
2. Yu. A. Malkov, A. Ponomarenko, A. Logvinov, V. Krylov. *Approximate nearest neighbor algorithm based on navigable small world graphs.* Information Systems, 2014.
3. M. Aumüller, E. Bernhardsson, A. Faithfull. *ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.* Information Systems, 2020. [ann-benchmarks.com](https://ann-benchmarks.com)

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
