#include "benchmark.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace hnsw {
namespace {

std::size_t queryCount(const VectorSet& queries, std::size_t maxQueries) {
    if (maxQueries == 0) return queries.count;
    return std::min<std::size_t>(maxQueries, queries.count);
}

/// Turns a vector of per-query latencies into a report row.
BenchmarkRow summarize(std::size_t ef, double recall,
                       const std::vector<double>& latenciesMs,
                       double totalSeconds) {
    BenchmarkRow row;
    row.ef = ef;
    row.recall = recall;
    row.medianMs = median(latenciesMs);
    row.p95Ms = percentile(latenciesMs, 95.0);
    row.qps = totalSeconds > 0.0
                  ? static_cast<double>(latenciesMs.size()) / totalSeconds
                  : 0.0;
    return row;
}

}  // namespace

std::vector<BenchmarkRow> runBenchmark(const HnswIndex& index,
                                       const VectorSet& queries,
                                       const GroundTruth& groundTruth,
                                       const BenchmarkConfig& config) {
    if (queries.dim != index.dim()) {
        throw std::invalid_argument("benchmark: query dimension mismatch");
    }
    const std::size_t numQueries = queryCount(queries, config.maxQueries);
    if (numQueries == 0) throw std::invalid_argument("benchmark: no queries");

    const bool measureRecall = !groundTruth.empty();
    if (measureRecall && numQueries > groundTruth.numQueries) {
        throw std::invalid_argument(
            "benchmark: ground truth covers fewer queries than requested");
    }

    // One scratch context reused across every query: the timings then reflect
    // graph traversal, not allocator behaviour.
    HnswIndex::SearchContext context;

    std::vector<BenchmarkRow> rows;
    rows.reserve(config.efValues.size());

    for (std::size_t ef : config.efValues) {
        // Warm up: touch the vector store and the graph before timing.
        const std::size_t warmup = std::min(config.warmupQueries, numQueries);
        for (std::size_t i = 0; i < warmup; ++i) {
            index.searchKnn(queries.at(i), config.k, ef, context);
        }

        std::vector<double> latencies;
        latencies.reserve(numQueries);
        std::vector<std::vector<std::uint32_t>> predictions;
        predictions.reserve(numQueries);

        const Timer wall;
        for (std::size_t i = 0; i < numQueries; ++i) {
            const Timer query;
            const std::vector<SearchResult> found =
                index.searchKnn(queries.at(i), config.k, ef, context);
            latencies.push_back(query.elapsedMillis());
            predictions.push_back(toIds(found));
        }
        const double seconds = wall.elapsedSeconds();

        const double recall =
            measureRecall ? recallAtK(predictions, groundTruth, config.k) : 0.0;
        rows.push_back(summarize(ef, recall, latencies, seconds));

        if (config.verbose) {
            std::printf("  ef=%-5zu recall@%zu=%.4f  median=%.3f ms  p95=%.3f ms  %.0f qps\n",
                        ef, config.k, rows.back().recall, rows.back().medianMs,
                        rows.back().p95Ms, rows.back().qps);
            std::fflush(stdout);
        }
    }
    return rows;
}

BenchmarkRow benchmarkBruteForce(const BruteForceIndex& index,
                                 const VectorSet& queries,
                                 const GroundTruth& groundTruth, std::size_t k,
                                 std::size_t maxQueries) {
    const std::size_t numQueries = queryCount(queries, maxQueries);
    if (numQueries == 0) throw std::invalid_argument("benchmark: no queries");

    std::vector<double> latencies;
    latencies.reserve(numQueries);
    std::vector<std::vector<std::uint32_t>> predictions;
    predictions.reserve(numQueries);

    const Timer wall;
    for (std::size_t i = 0; i < numQueries; ++i) {
        const Timer query;
        const std::vector<SearchResult> found = index.search(queries.at(i), k);
        latencies.push_back(query.elapsedMillis());
        predictions.push_back(toIds(found));
    }
    const double seconds = wall.elapsedSeconds();

    // Recall is 1.0 by construction; it is recomputed anyway so that a broken
    // ground truth file shows up immediately.
    const double recall =
        groundTruth.empty() ? 1.0 : recallAtK(predictions, groundTruth, k);
    return summarize(0, recall, latencies, seconds);
}

void writeCsv(const std::string& path, const std::vector<BenchmarkRow>& rows) {
    ensureParentDirectory(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("benchmark: cannot write '" + path + "'");

    out << "ef,recall,median_ms,p95_ms,qps\n";
    char buffer[256];
    for (const BenchmarkRow& row : rows) {
        std::snprintf(buffer, sizeof(buffer), "%zu,%.6f,%.6f,%.6f,%.2f\n",
                      row.ef, row.recall, row.medianMs, row.p95Ms, row.qps);
        out << buffer;
    }
    if (!out) throw std::runtime_error("benchmark: failed to write '" + path + "'");
}

void printBenchmarkTable(const std::vector<BenchmarkRow>& rows) {
    std::printf("\n%8s %10s %12s %12s %12s\n", "ef", "recall", "median_ms",
                "p95_ms", "qps");
    std::printf("%8s %10s %12s %12s %12s\n", "--------", "----------",
                "------------", "------------", "------------");
    for (const BenchmarkRow& row : rows) {
        std::printf("%8zu %10.4f %12.3f %12.3f %12.1f\n", row.ef, row.recall,
                    row.medianMs, row.p95Ms, row.qps);
    }
    std::printf("\n");
}

}  // namespace hnsw
