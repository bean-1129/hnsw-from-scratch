// hnsw.hpp -- Hierarchical Navigable Small World index.
//
// Implementation of
//   Yu. A. Malkov, D. A. Yashunin,
//   "Efficient and Robust Approximate Nearest Neighbor Search Using
//    Hierarchical Navigable Small World Graphs", TPAMI 2018 (arXiv:1603.09320)
//
// Every algorithm of the paper is implemented here, with the same names:
//
//   Algorithm 1  INSERT                     -> HnswIndex::addPoint
//   Algorithm 2  SEARCH-LAYER               -> HnswIndex::searchLayer
//   Algorithm 3  SELECT-NEIGHBORS-SIMPLE    -> HnswIndex::selectNeighborsSimple
//   Algorithm 4  SELECT-NEIGHBORS-HEURISTIC -> HnswIndex::selectNeighborsHeuristic
//   Algorithm 5  K-NN-SEARCH                -> HnswIndex::searchKnn
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "distance.hpp"
#include "graph.hpp"
#include "utils.hpp"
#include "vector_loader.hpp"

namespace hnsw {

/// A node id together with its distance to the query / base element.
struct Candidate {
    float distance = 0.0f;
    std::uint32_t id = kInvalidId;
};

/// Heap orderings. `std::push_heap` builds a max-heap w.r.t. the comparator,
/// so `CloserFirst` (which reports "greater distance = smaller") yields a
/// min-heap whose top is the closest element.
struct CloserFirst {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.distance > b.distance;
    }
};
struct FartherFirst {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.distance < b.distance;
    }
};

/// Construction and default query parameters.
struct HnswParams {
    /// Number of bidirectional links created per element per layer.
    /// Layer 0 uses `2 * M` (Mmax0 in the paper, section 4.1).
    std::size_t M = 16;
    /// Size of the dynamic candidate list during construction.
    std::size_t efConstruction = 200;
    /// Default size of the dynamic candidate list at query time.
    std::size_t efSearch = 64;
    /// Seed of the level generator; fixing it makes builds reproducible.
    std::uint64_t seed = 100;
    Metric metric = Metric::Cosine;
    /// Algorithm 4, `extendCandidates`. Off by default, as recommended by the
    /// paper for datasets without pronounced clustering.
    bool extendCandidates = false;
    /// Algorithm 4, `keepPrunedConnections`. On by default: it keeps the
    /// out-degree at Mmax and improves recall on high-dimensional data.
    bool keepPrunedConnections = true;
    /// Level normalization factor mL. When <= 0 the paper's optimum
    /// `1 / ln(M)` is used.
    double levelMultiplier = 0.0;
};

/// Aggregate statistics used by the CLI and the benchmark report.
struct IndexStats {
    std::size_t numElements = 0;
    std::uint32_t dim = 0;
    int maxLevel = -1;
    std::size_t totalEdges = 0;
    std::vector<std::size_t> nodesPerLayer;
    std::vector<double> averageDegreePerLayer;
    std::size_t vectorBytes = 0;
    std::size_t graphBytes = 0;
};

class HnswIndex {
public:
    /// Scratch space reused across queries so that a search performs no
    /// allocation after the first call. One context per thread.
    struct SearchContext {
        VisitedList visited;
        std::vector<Candidate> candidates;  ///< min-heap C of the paper
        std::vector<Candidate> results;     ///< max-heap W of the paper

        void prepare(std::size_t numElements) {
            visited.resize(numElements);
            visited.reset();
            candidates.clear();
            results.clear();
        }
    };

    /// Progress callback invoked by `build` as (inserted, total).
    using ProgressCallback = std::function<void(std::size_t, std::size_t)>;

    HnswIndex(std::uint32_t dim, const HnswParams& params);

    // -- construction -------------------------------------------------------

    /// Reserves storage for `n` elements. Optional, but it avoids reallocating
    /// the vector store while inserting a large dataset.
    void reserve(std::size_t n);

    /// Algorithm 1: inserts one element and returns its id (ids are handed out
    /// consecutively, starting at 0). `vector` is copied into the index and
    /// must already be normalized when the metric is cosine.
    std::uint32_t addPoint(const float* vector);

    /// Inserts every row of `vectors`, reporting progress every
    /// `progressEvery` insertions (0 disables the callback).
    void build(const VectorSet& vectors, const ProgressCallback& onProgress = {},
               std::size_t progressEvery = 10000);

    // -- query --------------------------------------------------------------

    /// Algorithm 5, using a thread-local scratch context and the default
    /// `efSearch` from the parameters.
    std::vector<SearchResult> searchKnn(const float* query,
                                        std::size_t k) const;

    /// Algorithm 5 with an explicit `ef`. `ef` is raised to `k` when smaller,
    /// as the beam must be at least as wide as the requested result set.
    std::vector<SearchResult> searchKnn(const float* query, std::size_t k,
                                        std::size_t ef) const;

    /// Overload taking a caller-owned context; use it to run concurrent
    /// queries from several threads without any shared mutable state.
    std::vector<SearchResult> searchKnn(const float* query, std::size_t k,
                                        std::size_t ef,
                                        SearchContext& ctx) const;

    // -- accessors ----------------------------------------------------------

    const float* vectorAt(std::uint32_t id) const {
        return data_.data() + static_cast<std::size_t>(id) * dim_;
    }
    std::size_t size() const { return graph_.size(); }
    std::uint32_t dim() const { return dim_; }
    int maxLevel() const { return maxLevel_; }
    std::uint32_t entryPoint() const { return entryPoint_; }
    const HnswParams& params() const { return params_; }
    HnswParams& mutableParams() { return params_; }
    const Graph& graph() const { return graph_; }
    Metric metric() const { return params_.metric; }

    IndexStats stats() const;

    // -- persistence --------------------------------------------------------

    /// Serializes parameters, vectors and topology to `path`.
    void save(const std::string& path) const;

    /// Reads back an index written by `save`.
    static HnswIndex load(const std::string& path);

private:
    // -- algorithms of the paper -------------------------------------------

    /// Algorithm 2 (SEARCH-LAYER). Explores `layer` starting from
    /// `entryPoints` and leaves the `ef` closest elements found in
    /// `ctx.results` (a max-heap ordered by distance).
    void searchLayer(const float* query,
                     const std::vector<std::uint32_t>& entryPoints,
                     std::size_t ef, int layer, SearchContext& ctx) const;

    /// Algorithm 3 (SELECT-NEIGHBORS-SIMPLE): the M closest candidates.
    std::vector<std::uint32_t> selectNeighborsSimple(
        std::vector<Candidate> candidates, std::size_t M) const;

    /// Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC): relative-neighborhood style
    /// pruning that keeps long-range links alive. `base` is the element the
    /// candidate distances were measured from.
    std::vector<std::uint32_t> selectNeighborsHeuristic(
        const float* base, std::vector<Candidate> candidates, std::size_t M,
        int layer, bool extendCandidates, bool keepPrunedConnections) const;

    /// Links `newId` to `selected` on `layer` and repairs the reverse edges,
    /// shrinking over-full neighborhoods with Algorithm 4.
    void connectNeighbors(std::uint32_t newId,
                          const std::vector<std::uint32_t>& selected, int layer);

    /// Random level from the exponentially decaying distribution of section 4.1:
    /// `floor(-ln(U(0,1)) * mL)`.
    int randomLevel();

    float distanceTo(const float* query, std::uint32_t id) const {
        return distance(query, vectorAt(id), dim_, params_.metric);
    }

    /// Per-thread scratch used by the convenience `searchKnn` overloads.
    SearchContext& threadContext() const;

    std::uint32_t dim_;
    HnswParams params_;
    /// mL of section 4.1. The degree caps Mmax / Mmax0 live in `graph_`.
    double levelMultiplier_;

    std::vector<float> data_;  ///< contiguous vector store, row major
    Graph graph_;
    std::uint32_t entryPoint_ = kInvalidId;
    int maxLevel_ = -1;

    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

}  // namespace hnsw
