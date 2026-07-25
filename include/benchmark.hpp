// benchmark.hpp -- efSearch sweep producing the CSV consumed by the README
// plots and by the FAISS comparison script.
//
// Latency is measured per single query (batch size 1, one thread), which is the
// regime an online serving system actually runs in. QPS is the reciprocal of
// the mean query time, i.e. single-threaded throughput.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "brute_force.hpp"
#include "hnsw.hpp"
#include "recall.hpp"
#include "vector_loader.hpp"

namespace hnsw {

/// One row of the report: `ef,recall,median_ms,p95_ms,qps`.
struct BenchmarkRow {
    std::size_t ef = 0;
    double recall = 0.0;
    double medianMs = 0.0;
    double p95Ms = 0.0;
    double qps = 0.0;
};

struct BenchmarkConfig {
    std::vector<std::size_t> efValues{16, 32, 64, 128, 256, 512};
    std::size_t k = 10;
    /// 0 = use every query in the file.
    std::size_t maxQueries = 0;
    /// Queries issued before timing starts, to warm caches and branch
    /// predictors. Never counted in the reported numbers.
    std::size_t warmupQueries = 100;
    bool verbose = true;
};

/// Runs the sweep. Recall is evaluated against `groundTruth`; pass an empty
/// ground truth to time the index without measuring accuracy.
std::vector<BenchmarkRow> runBenchmark(const HnswIndex& index,
                                       const VectorSet& queries,
                                       const GroundTruth& groundTruth,
                                       const BenchmarkConfig& config);

/// Times the exact scan on the same queries, for the "speedup over brute
/// force" column of the README. `ef` is reported as 0 and recall as 1.0.
BenchmarkRow benchmarkBruteForce(const BruteForceIndex& index,
                                 const VectorSet& queries,
                                 const GroundTruth& groundTruth,
                                 std::size_t k, std::size_t maxQueries = 0);

/// Writes `ef,recall,median_ms,p95_ms,qps` including the header line.
void writeCsv(const std::string& path, const std::vector<BenchmarkRow>& rows);

/// Pretty-prints the same rows to stdout.
void printBenchmarkTable(const std::vector<BenchmarkRow>& rows);

}  // namespace hnsw
