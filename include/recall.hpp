// recall.hpp -- Recall@k evaluation.
//
// Recall@k for a single query is
//
//     |{predicted top-k} ∩ {true top-k}| / k
//
// and the reported number is the mean over all queries. Ties in the ground
// truth are ignored on purpose: this is the metric ann-benchmarks reports, so
// the numbers here are directly comparable to published results.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "utils.hpp"
#include "vector_loader.hpp"

namespace hnsw {

/// Recall of one query. `truth` must point at (at least) `k` ground truth ids.
double recallForQuery(const std::vector<std::uint32_t>& predicted,
                      const std::int32_t* truth, std::size_t k);

/// Mean Recall@k over all queries.
/// `predicted.size()` must not exceed the number of ground truth rows, which
/// makes it legal to evaluate a subset of the query file.
double recallAtK(const std::vector<std::vector<std::uint32_t>>& predicted,
                 const GroundTruth& groundTruth, std::size_t k);

/// Overload for results that still carry distances.
double recallAtK(const std::vector<std::vector<SearchResult>>& predicted,
                 const GroundTruth& groundTruth, std::size_t k);

/// Drops the distances from a result list.
std::vector<std::uint32_t> toIds(const std::vector<SearchResult>& results);

}  // namespace hnsw
