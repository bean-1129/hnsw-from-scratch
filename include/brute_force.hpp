// brute_force.hpp -- exact k-NN by full linear scan.
//
// This is the correctness oracle for the whole project: the HNSW recall
// numbers, the unit tests and the ground truth generator all compare against
// it. It is intentionally simple; the only concession to speed is an optional
// multi-threaded batch mode used when regenerating ground truth.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "distance.hpp"
#include "utils.hpp"
#include "vector_loader.hpp"

namespace hnsw {

class BruteForceIndex {
public:
    /// The index keeps a *reference* to the caller's vectors; `vectors` must
    /// outlive the index.
    BruteForceIndex(const VectorSet& vectors, Metric metric = Metric::Cosine);

    /// Exact `k` nearest neighbors of `query`, sorted by increasing distance.
    std::vector<SearchResult> search(const float* query, std::size_t k) const;

    /// Exact neighbor *ids* for every row of `queries`.
    /// `numThreads == 0` selects `std::thread::hardware_concurrency()`.
    std::vector<std::vector<std::uint32_t>> searchBatch(
        const VectorSet& queries, std::size_t k,
        std::size_t numThreads = 0) const;

    /// Runs `searchBatch` and packs the result into a `GroundTruth` blob.
    GroundTruth computeGroundTruth(const VectorSet& queries, std::size_t k,
                                   std::size_t numThreads = 0) const;

    std::size_t size() const { return count_; }
    std::uint32_t dim() const { return dim_; }
    Metric metric() const { return metric_; }

private:
    const float* data_;
    std::size_t count_;
    std::uint32_t dim_;
    Metric metric_;
};

}  // namespace hnsw
