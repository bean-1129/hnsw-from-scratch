#include "hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace hnsw {
namespace {

constexpr char kIndexMagic[8] = {'H', 'N', 'S', 'W', 'I', 'D', 'X', '\0'};
constexpr std::uint32_t kIndexFormatVersion = 1;

/// Pushes onto a heap kept in a plain vector (so it can be cleared without
/// deallocating, unlike std::priority_queue).
template <typename Compare>
void heapPush(std::vector<Candidate>& heap, const Candidate& value,
              Compare compare) {
    heap.push_back(value);
    std::push_heap(heap.begin(), heap.end(), compare);
}

template <typename Compare>
Candidate heapPop(std::vector<Candidate>& heap, Compare compare) {
    std::pop_heap(heap.begin(), heap.end(), compare);
    const Candidate top = heap.back();
    heap.pop_back();
    return top;
}

/// Element with the smallest distance in an arbitrary (unsorted) container.
Candidate nearestOf(const std::vector<Candidate>& candidates) {
    Candidate best{std::numeric_limits<float>::max(), kInvalidId};
    for (const Candidate& c : candidates) {
        if (c.distance < best.distance) best = c;
    }
    return best;
}

}  // namespace

HnswIndex::HnswIndex(std::uint32_t dim, const HnswParams& params)
    : dim_(dim),
      params_(params),
      levelMultiplier_(params.levelMultiplier),
      // Section 4.1: layer 0 keeps twice as many links (Mmax0) as the layers
      // above it (Mmax = M).
      graph_(params.M, params.M * 2),
      rng_(params.seed) {
    if (dim_ == 0) throw std::invalid_argument("hnsw: dimension must be positive");
    if (params_.M < 2) throw std::invalid_argument("hnsw: M must be at least 2");
    if (params_.efConstruction == 0) {
        throw std::invalid_argument("hnsw: efConstruction must be positive");
    }
    if (levelMultiplier_ <= 0.0) {
        // mL = 1 / ln(M) minimizes the overlap between layers (section 4.1).
        levelMultiplier_ = 1.0 / std::log(static_cast<double>(params_.M));
    }
}

// ---------------------------------------------------------------------------
// Level generation
// ---------------------------------------------------------------------------

int HnswIndex::randomLevel() {
    double u = uniform_(rng_);
    // The distribution is defined on (0, 1]; guard against log(0).
    if (u <= 0.0) u = std::numeric_limits<double>::min();
    const double level = -std::log(u) * levelMultiplier_;
    return static_cast<int>(std::floor(level));
}

// ---------------------------------------------------------------------------
// Algorithm 2 -- SEARCH-LAYER
// ---------------------------------------------------------------------------
//
//   C <- ep (min-heap by distance), W <- ep (max-heap by distance)
//   while |C| > 0:
//       c <- nearest element of C;  f <- furthest element of W
//       if dist(c, q) > dist(f, q): break        // all remaining are worse
//       for each e in neighbourhood(c, lc) not yet visited:
//           if dist(e, q) < dist(f, q) or |W| < ef:
//               C <- C u {e};  W <- W u {e}
//               if |W| > ef: remove furthest from W
//   return W
void HnswIndex::searchLayer(const float* query,
                            const std::vector<std::uint32_t>& entryPoints,
                            std::size_t ef, int layer,
                            SearchContext& ctx) const {
    ctx.visited.resize(graph_.size());
    ctx.visited.reset();  // a node visited on the layer above may be revisited
    ctx.candidates.clear();
    ctx.results.clear();

    for (std::uint32_t ep : entryPoints) {
        if (ep == kInvalidId || !ctx.visited.markVisited(ep)) continue;
        const Candidate seed{distanceTo(query, ep), ep};
        heapPush(ctx.candidates, seed, CloserFirst{});
        heapPush(ctx.results, seed, FartherFirst{});
    }
    // More entry points than the beam is wide: keep only the best `ef`.
    while (ctx.results.size() > ef) heapPop(ctx.results, FartherFirst{});

    while (!ctx.candidates.empty()) {
        const Candidate closest = heapPop(ctx.candidates, CloserFirst{});
        // `ctx.results.front()` is the furthest element currently kept.
        if (!ctx.results.empty() &&
            closest.distance > ctx.results.front().distance) {
            break;
        }

        const Graph::NeighborList& neighbors = graph_.neighbors(closest.id, layer);
        for (std::uint32_t neighbor : neighbors) {
            if (!ctx.visited.markVisited(neighbor)) continue;

            const float d = distanceTo(query, neighbor);
            if (ctx.results.size() < ef || d < ctx.results.front().distance) {
                const Candidate candidate{d, neighbor};
                heapPush(ctx.candidates, candidate, CloserFirst{});
                heapPush(ctx.results, candidate, FartherFirst{});
                if (ctx.results.size() > ef) {
                    heapPop(ctx.results, FartherFirst{});
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm 3 -- SELECT-NEIGHBORS-SIMPLE
// ---------------------------------------------------------------------------
std::vector<std::uint32_t> HnswIndex::selectNeighborsSimple(
    std::vector<Candidate> candidates, std::size_t M) const {
    const std::size_t keep = std::min(M, candidates.size());
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.distance < b.distance;
                      });
    std::vector<std::uint32_t> selected;
    selected.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) selected.push_back(candidates[i].id);
    return selected;
}

// ---------------------------------------------------------------------------
// Algorithm 4 -- SELECT-NEIGHBORS-HEURISTIC
// ---------------------------------------------------------------------------
//
// The point of the heuristic is *diversity*: a candidate e is kept only if it
// is closer to the base element than to any already selected neighbor. That
// prevents every link of a node from pointing into the same dense cluster and
// preserves the long-range edges that make the graph navigable.
std::vector<std::uint32_t> HnswIndex::selectNeighborsHeuristic(
    const float* base, std::vector<Candidate> candidates, std::size_t M,
    int layer, bool extendCandidates, bool keepPrunedConnections) const {
    std::vector<std::uint32_t> selected;
    if (M == 0 || candidates.empty()) return selected;

    if (extendCandidates) {
        // W <- W u neighbourhood(e) for every candidate e.
        std::unordered_set<std::uint32_t> seen;
        seen.reserve(candidates.size() * 2);
        for (const Candidate& c : candidates) seen.insert(c.id);

        const std::size_t original = candidates.size();
        for (std::size_t i = 0; i < original; ++i) {
            const std::uint32_t id = candidates[i].id;
            for (std::uint32_t adjacent : graph_.neighbors(id, layer)) {
                if (seen.insert(adjacent).second) {
                    candidates.push_back({distanceTo(base, adjacent), adjacent});
                }
            }
        }
    }

    // W: working queue, ordered by increasing distance to the base element.
    std::make_heap(candidates.begin(), candidates.end(), CloserFirst{});
    std::vector<Candidate> discarded;  // Wd, filled in increasing distance order
    selected.reserve(M);

    while (!candidates.empty() && selected.size() < M) {
        const Candidate e = heapPop(candidates, CloserFirst{});
        const float* candidateVector = vectorAt(e.id);

        bool closerToBase = true;
        for (std::uint32_t chosen : selected) {
            const float toChosen = distance(candidateVector, vectorAt(chosen),
                                            dim_, params_.metric);
            if (toChosen < e.distance) {
                closerToBase = false;  // e sits "behind" an already chosen link
                break;
            }
        }

        if (closerToBase) {
            selected.push_back(e.id);
        } else if (keepPrunedConnections) {
            discarded.push_back(e);
        }
    }

    // Top up with the best pruned candidates so the node reaches its full
    // out-degree. `discarded` is already sorted by increasing distance because
    // it was filled from a min-heap.
    if (keepPrunedConnections) {
        for (const Candidate& c : discarded) {
            if (selected.size() >= M) break;
            selected.push_back(c.id);
        }
    }
    return selected;
}

// ---------------------------------------------------------------------------
// Bidirectional linking (second half of Algorithm 1)
// ---------------------------------------------------------------------------
void HnswIndex::connectNeighbors(std::uint32_t newId,
                                 const std::vector<std::uint32_t>& selected,
                                 int layer) {
    graph_.setNeighbors(newId, layer, selected);

    const std::size_t maxDegree = graph_.maxDegree(layer);
    for (std::uint32_t neighbor : selected) {
        if (neighbor == newId) continue;

        // Fast path: the neighborhood still has room for the reverse edge.
        if (graph_.tryAddNeighbor(neighbor, layer, newId)) continue;

        // Otherwise shrink eConn u {newId} back to Mmax with Algorithm 4,
        // measuring distances from the *neighbor*, not from the new element.
        const float* neighborVector = vectorAt(neighbor);
        const Graph::NeighborList& connections = graph_.neighbors(neighbor, layer);

        std::vector<Candidate> candidates;
        candidates.reserve(connections.size() + 1);
        for (std::uint32_t existing : connections) {
            candidates.push_back(
                {distanceTo(neighborVector, existing), existing});
        }
        candidates.push_back({distanceTo(neighborVector, newId), newId});

        std::vector<std::uint32_t> pruned = selectNeighborsHeuristic(
            neighborVector, std::move(candidates), maxDegree, layer,
            /*extendCandidates=*/false, params_.keepPrunedConnections);
        graph_.setNeighbors(neighbor, layer, std::move(pruned));
    }
}

// ---------------------------------------------------------------------------
// Algorithm 1 -- INSERT
// ---------------------------------------------------------------------------
std::uint32_t HnswIndex::addPoint(const float* vector) {
    // l <- floor(-ln(unif(0,1)) * mL)
    const int level = randomLevel();

    const std::uint32_t id = graph_.addNode(level);
    data_.insert(data_.end(), vector, vector + dim_);

    // First element: it becomes the entry point and there is nothing to link to.
    if (entryPoint_ == kInvalidId) {
        entryPoint_ = id;
        maxLevel_ = level;
        return id;
    }

    SearchContext& ctx = threadContext();
    const float* query = vectorAt(id);
    std::vector<std::uint32_t> entryPoints{entryPoint_};

    // Phase 1: greedy descent (ef = 1) from the top layer down to level + 1.
    for (int layer = maxLevel_; layer > level; --layer) {
        searchLayer(query, entryPoints, 1, layer, ctx);
        if (ctx.results.empty()) break;
        entryPoints.assign(1, nearestOf(ctx.results).id);
    }

    // Phase 2: from min(L, l) down to 0, insert links on every layer.
    const int startLayer = std::min(maxLevel_, level);
    for (int layer = startLayer; layer >= 0; --layer) {
        searchLayer(query, entryPoints, params_.efConstruction, layer, ctx);
        const std::vector<Candidate> found = ctx.results;

        const std::vector<std::uint32_t> neighbors = selectNeighborsHeuristic(
            query, found, params_.M, layer, params_.extendCandidates,
            params_.keepPrunedConnections);
        connectNeighbors(id, neighbors, layer);

        // "ep <- W": the whole candidate set seeds the search one layer down.
        entryPoints.clear();
        entryPoints.reserve(found.size());
        for (const Candidate& c : found) entryPoints.push_back(c.id);
    }

    if (level > maxLevel_) {
        maxLevel_ = level;
        entryPoint_ = id;
    }
    return id;
}

void HnswIndex::reserve(std::size_t n) {
    data_.reserve(n * dim_);
}

void HnswIndex::build(const VectorSet& vectors,
                      const ProgressCallback& onProgress,
                      std::size_t progressEvery) {
    if (vectors.dim != dim_) {
        throw std::invalid_argument("hnsw: build() dimension mismatch (index " +
                                    std::to_string(dim_) + ", data " +
                                    std::to_string(vectors.dim) + ")");
    }
    reserve(size() + vectors.count);

    for (std::size_t i = 0; i < vectors.count; ++i) {
        addPoint(vectors.at(i));
        if (onProgress && progressEvery > 0 && (i + 1) % progressEvery == 0) {
            onProgress(i + 1, vectors.count);
        }
    }
    // Final tick, unless the loop above already reported the last insertion.
    const bool alreadyReported =
        progressEvery > 0 && vectors.count % progressEvery == 0;
    if (onProgress && vectors.count > 0 && !alreadyReported) {
        onProgress(vectors.count, vectors.count);
    }
}

// ---------------------------------------------------------------------------
// Algorithm 5 -- K-NN-SEARCH
// ---------------------------------------------------------------------------
std::vector<SearchResult> HnswIndex::searchKnn(const float* query,
                                               std::size_t k, std::size_t ef,
                                               SearchContext& ctx) const {
    if (k == 0 || entryPoint_ == kInvalidId) return {};

    ctx.prepare(graph_.size());
    std::vector<std::uint32_t> entryPoints{entryPoint_};

    // Layers L .. 1: greedy descent with a beam of one.
    for (int layer = maxLevel_; layer > 0; --layer) {
        searchLayer(query, entryPoints, 1, layer, ctx);
        if (ctx.results.empty()) break;
        entryPoints.assign(1, nearestOf(ctx.results).id);
    }

    // Layer 0: beam search with the full dynamic candidate list.
    searchLayer(query, entryPoints, std::max(ef, k), 0, ctx);

    std::vector<Candidate> found = ctx.results;
    const std::size_t keep = std::min(k, found.size());
    std::partial_sort(found.begin(),
                      found.begin() + static_cast<std::ptrdiff_t>(keep),
                      found.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.distance < b.distance;
                      });

    std::vector<SearchResult> results;
    results.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) {
        results.push_back({found[i].id, found[i].distance});
    }
    return results;
}

std::vector<SearchResult> HnswIndex::searchKnn(const float* query,
                                               std::size_t k,
                                               std::size_t ef) const {
    return searchKnn(query, k, ef, threadContext());
}

std::vector<SearchResult> HnswIndex::searchKnn(const float* query,
                                               std::size_t k) const {
    return searchKnn(query, k, params_.efSearch, threadContext());
}

HnswIndex::SearchContext& HnswIndex::threadContext() const {
    // One scratch buffer per thread: queries never share mutable state, so
    // several threads may search the same (const) index concurrently.
    static thread_local SearchContext context;
    return context;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
IndexStats HnswIndex::stats() const {
    IndexStats stats;
    stats.numElements = graph_.size();
    stats.dim = dim_;
    stats.maxLevel = maxLevel_;
    stats.totalEdges = graph_.totalEdges();
    stats.vectorBytes = data_.capacity() * sizeof(float);
    stats.graphBytes = graph_.memoryUsageBytes();
    for (int layer = 0; layer <= maxLevel_; ++layer) {
        stats.nodesPerLayer.push_back(graph_.nodesOnLayer(layer));
        stats.averageDegreePerLayer.push_back(graph_.averageDegree(layer));
    }
    return stats;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
void HnswIndex::save(const std::string& path) const {
    ensureParentDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("hnsw: cannot write '" + path + "'");

    writeArray(out, kIndexMagic, sizeof(kIndexMagic));
    writePod(out, kIndexFormatVersion);
    writePod(out, dim_);
    writePod(out, static_cast<std::uint32_t>(params_.metric));
    writePod(out, static_cast<std::uint64_t>(params_.M));
    writePod(out, static_cast<std::uint64_t>(params_.efConstruction));
    writePod(out, static_cast<std::uint64_t>(params_.efSearch));
    writePod(out, params_.seed);
    writePod(out, static_cast<std::uint8_t>(params_.extendCandidates));
    writePod(out, static_cast<std::uint8_t>(params_.keepPrunedConnections));
    writePod(out, levelMultiplier_);
    writePod(out, static_cast<std::uint64_t>(graph_.size()));
    writePod(out, entryPoint_);
    writePod(out, static_cast<std::int32_t>(maxLevel_));

    writeArray(out, data_.data(), data_.size());
    graph_.save(out);
}

HnswIndex HnswIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("hnsw: cannot open '" + path + "'");

    char magic[sizeof(kIndexMagic)];
    readArray(in, magic, sizeof(magic));
    if (std::memcmp(magic, kIndexMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("hnsw: '" + path + "' is not an HNSW index");
    }

    std::uint32_t version = 0;
    readPod(in, version);
    if (version != kIndexFormatVersion) {
        throw std::runtime_error("hnsw: unsupported index version " +
                                 std::to_string(version));
    }

    std::uint32_t dim = 0;
    std::uint32_t metric = 0;
    std::uint64_t M = 0, efConstruction = 0, efSearch = 0;
    std::uint8_t extendCandidates = 0, keepPruned = 0;
    double levelMultiplier = 0.0;
    std::uint64_t numElements = 0;
    std::uint32_t entryPoint = kInvalidId;
    std::int32_t maxLevel = -1;

    readPod(in, dim);
    readPod(in, metric);
    HnswParams params;
    readPod(in, M);
    readPod(in, efConstruction);
    readPod(in, efSearch);
    readPod(in, params.seed);
    readPod(in, extendCandidates);
    readPod(in, keepPruned);
    readPod(in, levelMultiplier);
    readPod(in, numElements);
    readPod(in, entryPoint);
    readPod(in, maxLevel);

    params.metric = static_cast<Metric>(metric);
    params.M = static_cast<std::size_t>(M);
    params.efConstruction = static_cast<std::size_t>(efConstruction);
    params.efSearch = static_cast<std::size_t>(efSearch);
    params.extendCandidates = extendCandidates != 0;
    params.keepPrunedConnections = keepPruned != 0;
    params.levelMultiplier = levelMultiplier;

    HnswIndex index(dim, params);
    index.data_.resize(static_cast<std::size_t>(numElements) * dim);
    readArray(in, index.data_.data(), index.data_.size());
    index.graph_.load(in);
    index.entryPoint_ = entryPoint;
    index.maxLevel_ = maxLevel;

    if (index.graph_.size() != numElements) {
        throw std::runtime_error("hnsw: element count does not match the graph");
    }
    return index;
}

}  // namespace hnsw
