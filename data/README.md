# Datasets

This directory holds the binary files the C++ engine reads. Nothing here is
committed to git (see `.gitignore`) — the files are generated from the
ann-benchmarks HDF5 archive.

## Reference dataset

**glove-100-angular** — 100-dimensional GloVe word embeddings, angular (cosine)
distance.

| Property | Value |
|---|---|
| Base vectors | 1,183,514 |
| Dimension | 100 |
| Queries | 10,000 |
| Ground truth | 100 neighbors per query |
| Distance | angular / cosine |
| Source | <https://ann-benchmarks.com> |
| Direct URL | <http://ann-benchmarks.com/glove-100-angular.hdf5> |
| Download size | ≈ 460 MB |

## Generating the binaries

```bash
pip install numpy h5py
python ../python/convert_hdf5.py --download --output-dir .
```

or, if the archive is already on disk:

```bash
python ../python/convert_hdf5.py --input glove-100-angular.hdf5 --output-dir .
```

Files produced:

| File | Contents | Approx. size |
|---|---|---|
| `base.fbin` | all 1,183,514 base vectors | 474 MB |
| `query.fbin` | all 10,000 query vectors | 4 MB |
| `groundtruth.ibin` | exact top-100 neighbors of each query, over the **full** base set | 4 MB |
| `subset100k.fbin` | the first 100,000 base vectors, for fast iteration | 40 MB |
| `groundtruth_subset100k.ibin` | exact top-100 neighbors **recomputed against the subset** | 4 MB |

### Why the subset gets its own ground truth

`groundtruth.ibin` stores ids into the full 1.18M-vector base set. Most of those
ids do not exist in a 100,000-vector subset, and the ones that do are usually not
the subset's true nearest neighbors. Evaluating recall on `subset100k.fbin`
against `groundtruth.ibin` therefore produces numbers that look catastrophically
bad for reasons that have nothing to do with the index.

`convert_hdf5.py` recomputes the exact neighbors for the subset with a blocked
matrix multiplication (pass `--no-subset-groundtruth` to skip it). You can also
regenerate any ground truth with the C++ tool, which is exact and multi-threaded:

```bash
../build/bin/hnsw bruteforce \
    --base subset100k.fbin --query query.fbin \
    --k 100 --out groundtruth_subset100k.ibin
```

## Troubleshooting the download

| Symptom | Cause | Status |
|---|---|---|
| `command not found: python` | many systems only provide `python3` | use `python3 python/convert_hdf5.py …` |
| `HTTP Error 403: Forbidden` | the CDN in front of ann-benchmarks.com rejects the default `Python-urllib/x.y` User-Agent | handled — the script sends its own User-Agent |
| `CERTIFICATE_VERIFY_FAILED` | a python.org macOS install ships without a CA bundle until `Install Certificates.command` is run | handled — the script falls back to `certifi` or the system store at `/etc/ssl/cert.pem`, without ever disabling verification |
| `ModuleNotFoundError: No module named 'h5py'` | the conversion step needs h5py (the download does not) | run `python3 -m pip install h5py` |

An interrupted download leaves no `.hdf5` behind: bytes go to a `.part` file that
is renamed only after the full `Content-Length` has arrived, so a partial
transfer can never be mistaken for a complete archive on the next run.

## File formats

Both formats are flat, header-prefixed and written in native (little endian)
byte order. The C++ side never reads HDF5.

`*.fbin` — vectors:

```
uint32   number_of_vectors
uint32   dimension
float32  vectors[number_of_vectors * dimension]     row major
```

`*.ibin` — neighbor ids:

```
uint32   number_of_queries
uint32   k
int32    neighbors[number_of_queries * k]           row major
```

Vectors are stored **un-normalized**, exactly as they appear in the archive.
Normalization happens once inside the C++ loader (`loadVectors`), which is what
makes `cosine distance == 1 - dot` valid throughout the search.

## Using your own data

Any `.fbin` file with the header above works. A minimal writer:

```python
import numpy as np

def write_fbin(path, vectors):
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(vectors.shape, dtype=np.uint32).tofile(f)
        vectors.tofile(f)
```

Then build an index and generate matching ground truth:

```bash
../build/bin/hnsw bruteforce --base my_base.fbin --query my_query.fbin --k 100 --out my_gt.ibin
../build/bin/hnsw build --base my_base.fbin --out ../results/my_index.hnsw --M 16 --ef-construction 200
```

For a Euclidean dataset, pass `--metric l2` to every command; the loader then
skips normalization and the index ranks by squared L2 distance.
