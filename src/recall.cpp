#include "recall.hpp"

#include <algorithm>
#include <stdexcept>

namespace hnsw {

std::vector<std::uint32_t> toIds(const std::vector<SearchResult>& results) {
    std::vector<std::uint32_t> ids;
    ids.reserve(results.size());
    for (const SearchResult& r : results) ids.push_back(r.id);
    return ids;
}

double recallForQuery(const std::vector<std::uint32_t>& predicted,
                      const std::int32_t* truth, std::size_t k) {
    if (k == 0) return 1.0;

    // k is small (10 by default), so a linear scan beats building a hash set:
    // it stays in registers/L1 and avoids an allocation per query.
    std::size_t hits = 0;
    const std::size_t limit = std::min(k, predicted.size());
    for (std::size_t i = 0; i < k; ++i) {
        const std::int32_t expected = truth[i];
        if (expected < 0) continue;  // padding in a short ground truth row
        for (std::size_t j = 0; j < limit; ++j) {
            if (predicted[j] == static_cast<std::uint32_t>(expected)) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

double recallAtK(const std::vector<std::vector<std::uint32_t>>& predicted,
                 const GroundTruth& groundTruth, std::size_t k) {
    if (predicted.empty()) return 0.0;
    if (groundTruth.empty()) {
        throw std::invalid_argument("recall: ground truth is empty");
    }
    if (predicted.size() > groundTruth.numQueries) {
        throw std::invalid_argument(
            "recall: more result rows than ground truth rows");
    }
    if (k > groundTruth.k) {
        throw std::invalid_argument("recall: ground truth holds only " +
                                    std::to_string(groundTruth.k) +
                                    " neighbors per query, need " +
                                    std::to_string(k));
    }

    double total = 0.0;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        total += recallForQuery(predicted[i], groundTruth.at(i), k);
    }
    return total / static_cast<double>(predicted.size());
}

double recallAtK(const std::vector<std::vector<SearchResult>>& predicted,
                 const GroundTruth& groundTruth, std::size_t k) {
    std::vector<std::vector<std::uint32_t>> ids;
    ids.reserve(predicted.size());
    for (const std::vector<SearchResult>& row : predicted) {
        ids.push_back(toIds(row));
    }
    return recallAtK(ids, groundTruth, k);
}

}  // namespace hnsw
