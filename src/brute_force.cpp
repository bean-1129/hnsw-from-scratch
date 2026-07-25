#include "brute_force.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>

namespace hnsw {

BruteForceIndex::BruteForceIndex(const VectorSet& vectors, Metric metric)
    : data_(vectors.data.data()),
      count_(vectors.count),
      dim_(vectors.dim),
      metric_(metric) {
    if (vectors.empty()) {
        throw std::invalid_argument("brute force: empty vector set");
    }
}

std::vector<SearchResult> BruteForceIndex::search(const float* query,
                                                  std::size_t k) const {
    const std::size_t topK = std::min(k, count_);
    if (topK == 0) return {};

    // Bounded max-heap: keep the k smallest distances seen so far, with the
    // current worst on top so it can be evicted in O(log k).
    std::vector<SearchResult> heap;
    heap.reserve(topK + 1);
    const auto worseFirst = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    };

    for (std::size_t i = 0; i < count_; ++i) {
        const float d = distance(query, data_ + i * dim_, dim_, metric_);
        if (heap.size() < topK) {
            heap.push_back({static_cast<std::uint32_t>(i), d});
            std::push_heap(heap.begin(), heap.end(), worseFirst);
        } else if (d < heap.front().distance) {
            std::pop_heap(heap.begin(), heap.end(), worseFirst);
            heap.back() = {static_cast<std::uint32_t>(i), d};
            std::push_heap(heap.begin(), heap.end(), worseFirst);
        }
    }

    std::sort_heap(heap.begin(), heap.end(), worseFirst);  // ascending distance
    return heap;
}

std::vector<std::vector<std::uint32_t>> BruteForceIndex::searchBatch(
    const VectorSet& queries, std::size_t k, std::size_t numThreads) const {
    if (queries.dim != dim_) {
        throw std::invalid_argument("brute force: query dimension mismatch");
    }

    std::vector<std::vector<std::uint32_t>> results(queries.count);
    if (queries.empty()) return results;

    if (numThreads == 0) {
        numThreads = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
    }
    numThreads = std::min<std::size_t>(numThreads, queries.count);

    // Dynamic work stealing over query indices: a static split would be uneven
    // whenever the machine is otherwise busy.
    std::atomic<std::size_t> next{0};
    const auto worker = [&]() {
        for (;;) {
            const std::size_t i = next.fetch_add(1);
            if (i >= queries.count) return;
            const std::vector<SearchResult> found = search(queries.at(i), k);
            std::vector<std::uint32_t>& row = results[i];
            row.reserve(found.size());
            for (const SearchResult& r : found) row.push_back(r.id);
        }
    };

    if (numThreads == 1) {
        worker();
        return results;
    }

    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    for (std::size_t t = 0; t < numThreads; ++t) pool.emplace_back(worker);
    for (std::thread& thread : pool) thread.join();
    return results;
}

GroundTruth BruteForceIndex::computeGroundTruth(const VectorSet& queries,
                                                std::size_t k,
                                                std::size_t numThreads) const {
    const std::size_t topK = std::min(k, count_);
    const std::vector<std::vector<std::uint32_t>> neighbors =
        searchBatch(queries, topK, numThreads);

    GroundTruth truth;
    truth.numQueries = queries.count;
    truth.k = static_cast<std::uint32_t>(topK);
    truth.neighbors.reserve(static_cast<std::size_t>(truth.numQueries) * topK);
    for (const std::vector<std::uint32_t>& row : neighbors) {
        for (std::uint32_t id : row) {
            truth.neighbors.push_back(static_cast<std::int32_t>(id));
        }
    }
    return truth;
}

}  // namespace hnsw
