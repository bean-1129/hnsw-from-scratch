// main.cpp -- command line front end of hnsw-from-scratch.
//
//   hnsw build      --base data/base.fbin --out results/index.hnsw
//   hnsw search     --index results/index.hnsw --query data/query.fbin --ef 64
//   hnsw benchmark  --index results/index.hnsw --query data/query.fbin \
//                   --gt data/groundtruth.ibin --csv results/hnsw_cpp.csv
//   hnsw save       --base data/base.fbin --out results/index.hnsw
//   hnsw load       --index results/index.hnsw
//   hnsw bruteforce --base data/base.fbin --query data/query.fbin --out gt.ibin
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "brute_force.hpp"
#include "distance.hpp"
#include "hnsw.hpp"
#include "recall.hpp"
#include "utils.hpp"
#include "vector_loader.hpp"

namespace {

using namespace hnsw;

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

/// Minimal `--key value` / `--key=value` / `--flag` parser.
class Args {
public:
    Args(int argc, char** argv, int firstIndex) {
        for (int i = firstIndex; i < argc; ++i) {
            std::string token = argv[i];
            if (token.rfind("--", 0) != 0) {
                throw std::invalid_argument("unexpected argument '" + token + "'");
            }
            token.erase(0, 2);

            const std::size_t equals = token.find('=');
            if (equals != std::string::npos) {
                values_[token.substr(0, equals)] = token.substr(equals + 1);
                continue;
            }
            // A bare "--flag" is stored as "1" so that `flag()` can see it, and
            // remembered as "bare" so that a numeric option cannot silently
            // read it as the value 1 -- that turns a quoting mistake such as
            // `--ef-construction $unset --k 10` into a wrong run instead of an
            // error.
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                values_[token] = argv[++i];
                bare_.erase(token);
            } else {
                values_[token] = "1";
                bare_.insert(token);
            }
        }
    }

    bool has(const std::string& key) const { return values_.count(key) > 0; }

    std::string str(const std::string& key, const std::string& fallback = "") const {
        const auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }

    std::string required(const std::string& key) const {
        const auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::invalid_argument("missing required option --" + key);
        }
        return it->second;
    }

    std::size_t size(const std::string& key, std::size_t fallback) const {
        return static_cast<std::size_t>(u64(key, fallback));
    }

    std::uint64_t u64(const std::string& key, std::uint64_t fallback) const {
        const auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        if (bare_.count(key) > 0) {
            throw std::invalid_argument("--" + key + " expects a value");
        }
        return parseUnsigned(key, it->second);
    }

    bool flag(const std::string& key) const {
        const auto it = values_.find(key);
        if (it == values_.end()) return false;
        return it->second != "0" && it->second != "false";
    }

    /// Rejects typos instead of silently ignoring them.
    void validate(const std::set<std::string>& known) const {
        for (const auto& entry : values_) {
            if (known.count(entry.first) == 0) {
                throw std::invalid_argument("unknown option --" + entry.first);
            }
        }
    }

private:
    /// Strict integer parsing. `std::stoull` alone would accept "16 200" as 16
    /// and "8abc" as 8, silently turning a quoting mistake in a benchmark
    /// script into a run with the wrong parameters.
    static std::uint64_t parseUnsigned(const std::string& key,
                                       const std::string& text) {
        std::size_t consumed = 0;
        std::uint64_t value = 0;
        try {
            value = std::stoull(text, &consumed);
        } catch (const std::exception&) {
            throw std::invalid_argument("--" + key + " expects a non-negative "
                                        "integer, got '" + text + "'");
        }
        if (consumed != text.size() || text.find('-') != std::string::npos) {
            throw std::invalid_argument("--" + key + " expects a non-negative "
                                        "integer, got '" + text + "'");
        }
        return value;
    }

    std::map<std::string, std::string> values_;
    std::set<std::string> bare_;  ///< options that appeared without a value
};

const std::set<std::string> kCommonOptions = {
    "base", "query",  "gt",     "index",  "out",   "k",
    "M",    "ef-construction", "ef",     "ef-list", "seed",  "metric",
    "limit", "queries", "csv",  "threads", "extend-candidates",
    "no-keep-pruned", "quiet", "show", "brute-force", "verify"};

void printUsage() {
    std::cout << R"(hnsw-from-scratch -- HNSW (Malkov & Yashunin) implemented from scratch

Usage: hnsw <command> [options]

Commands:
  build       Build an index from a .fbin base file (optionally save it).
  search      Run k-NN queries against an index and report recall/latency.
  benchmark   Sweep efSearch and write ef,recall,median_ms,p95_ms,qps as CSV.
  save        Build an index and persist it to disk (build + --out).
  load        Load an index from disk and print its statistics.
  bruteforce  Exact search; writes a ground truth .ibin when --out is given.

Data options:
  --base <path>        Base vectors (.fbin).
  --query <path>       Query vectors (.fbin).
  --gt <path>          Ground truth neighbor ids (.ibin).
  --index <path>       Serialized index to load.
  --out <path>         Output path (index for build/save, .ibin for bruteforce).
  --csv <path>         Benchmark CSV output path.
  --limit <n>          Use only the first n base vectors.
  --queries <n>        Use only the first n queries.

Index options:
  --M <n>              Links per element per layer      (default 16).
  --ef-construction <n>  Candidate list size at build   (default 200).
  --ef <n>             Candidate list size at query     (default 64).
  --ef-list <a,b,c>    efSearch sweep for benchmark  (default 16,32,64,128,256,512).
  --k <n>              Neighbors to return              (default 10).
  --seed <n>           RNG seed for level generation    (default 100).
  --metric <name>      cosine (default) or l2.
  --extend-candidates  Enable Algorithm 4 extendCandidates.
  --no-keep-pruned     Disable Algorithm 4 keepPrunedConnections.

Misc:
  --threads <n>        Threads for brute force (0 = hardware concurrency).
  --brute-force        benchmark: also time the exact scan for comparison.
  --verify             build/load: sanity check recall against --gt.
  --show <n>           search: print the first n result rows (default 5).
  --quiet              Suppress progress output.
)";
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

HnswParams paramsFromArgs(const Args& args) {
    HnswParams params;
    params.M = args.size("M", 16);
    params.efConstruction = args.size("ef-construction", 200);
    params.efSearch = args.size("ef", 64);
    params.seed = args.u64("seed", 100);
    params.metric = metricFromString(args.str("metric", "cosine"));
    params.extendCandidates = args.flag("extend-candidates");
    params.keepPrunedConnections = !args.flag("no-keep-pruned");
    return params;
}

void printStats(const HnswIndex& index) {
    const IndexStats stats = index.stats();
    std::printf("Index: %zu elements, dim %u, metric %s\n", stats.numElements,
                stats.dim, metricName(index.metric()));
    std::printf("  M=%zu  efConstruction=%zu  seed=%llu  entry point=%u  top layer=%d\n",
                index.params().M, index.params().efConstruction,
                static_cast<unsigned long long>(index.params().seed),
                index.entryPoint(), stats.maxLevel);
    std::printf("  edges: %zu   memory: vectors %s + graph %s\n", stats.totalEdges,
                formatBytes(stats.vectorBytes).c_str(),
                formatBytes(stats.graphBytes).c_str());
    for (std::size_t layer = 0; layer < stats.nodesPerLayer.size(); ++layer) {
        std::printf("  layer %-2zu: %9zu nodes, average degree %6.2f\n", layer,
                    stats.nodesPerLayer[layer], stats.averageDegreePerLayer[layer]);
    }
}

VectorSet loadBase(const Args& args, Metric metric, bool quiet) {
    const std::string path = args.required("base");
    const Timer timer;
    VectorSet base = loadVectors(path, metric, args.size("limit", 0));
    if (!quiet) {
        std::printf("Loaded %u x %u base vectors from %s (%.2f s, %s)\n",
                    base.count, base.dim, path.c_str(), timer.elapsedSeconds(),
                    formatBytes(base.data.size() * sizeof(float)).c_str());
    }
    return base;
}

VectorSet loadQueries(const Args& args, Metric metric, bool quiet) {
    const std::string path = args.required("query");
    VectorSet queries = loadVectors(path, metric, args.size("queries", 0));
    if (!quiet) {
        std::printf("Loaded %u x %u query vectors from %s\n", queries.count,
                    queries.dim, path.c_str());
    }
    return queries;
}

GroundTruth loadGroundTruthIfGiven(const Args& args, bool quiet) {
    if (!args.has("gt")) return {};
    GroundTruth truth = loadIbin(args.str("gt"));
    if (!quiet) {
        std::printf("Loaded ground truth: %u queries x %u neighbors\n",
                    truth.numQueries, truth.k);
    }
    return truth;
}

/// Builds an index over `base`, printing progress unless `quiet`.
HnswIndex buildIndex(const VectorSet& base, const HnswParams& params, bool quiet) {
    HnswIndex index(base.dim, params);
    index.reserve(base.count);

    const Timer timer;
    HnswIndex::ProgressCallback progress;
    if (!quiet) {
        progress = [&timer](std::size_t done, std::size_t total) {
            const double seconds = timer.elapsedSeconds();
            const double rate =
                seconds > 0.0 ? static_cast<double>(done) / seconds : 0.0;
            std::printf("\r  inserted %zu / %zu (%.1f%%)  %.0f vec/s  %.1f s elapsed   ",
                        done, total,
                        100.0 * static_cast<double>(done) / static_cast<double>(total),
                        rate, seconds);
            std::fflush(stdout);
        };
    }

    index.build(base, progress, 10000);
    if (!quiet) {
        std::printf("\nBuilt index in %.2f s (%.0f vectors/s)\n",
                    timer.elapsedSeconds(),
                    static_cast<double>(base.count) / timer.elapsedSeconds());
    }
    return index;
}

/// `--index` wins when both are present; otherwise the index is built from
/// `--base`. Every command that needs an index goes through here.
HnswIndex obtainIndex(const Args& args, bool quiet) {
    if (args.has("index")) {
        const Timer timer;
        HnswIndex index = HnswIndex::load(args.str("index"));
        if (args.has("ef")) index.mutableParams().efSearch = args.size("ef", 64);
        if (!quiet) {
            std::printf("Loaded index from %s (%.2f s)\n", args.str("index").c_str(),
                        timer.elapsedSeconds());
        }
        return index;
    }
    const HnswParams params = paramsFromArgs(args);
    const VectorSet base = loadBase(args, params.metric, quiet);
    return buildIndex(base, params, quiet);
}

/// Warns when a cosine index is fed data that is not unit length -- the single
/// most likely cause of unexpectedly poor recall.
void checkNormalization(const VectorSet& vectors, Metric metric,
                        const char* what) {
    if (metric != Metric::Cosine || vectors.empty()) return;
    const float deviation = vectors.head(1000).maxNormDeviation();
    if (deviation > 1e-3f) {
        std::fprintf(stderr,
                     "warning: %s vectors deviate from unit norm by %.4f; "
                     "cosine distance assumes normalized input\n",
                     what, deviation);
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

int commandBuild(const Args& args, bool requireOutput) {
    args.validate(kCommonOptions);

    const HnswParams params = paramsFromArgs(args);
    const bool quiet = args.flag("quiet");
    const VectorSet base = loadBase(args, params.metric, quiet);
    checkNormalization(base, params.metric, "base");

    HnswIndex index = buildIndex(base, params, quiet);
    printStats(index);

    const std::string out =
        requireOutput ? args.required("out") : args.str("out");
    if (!out.empty()) {
        const Timer timer;
        index.save(out);
        std::printf("Saved index to %s (%.2f s)\n", out.c_str(),
                    timer.elapsedSeconds());
    }

    if (args.flag("verify") && args.has("query") && args.has("gt")) {
        const VectorSet queries = loadQueries(args, params.metric, quiet);
        const GroundTruth truth = loadGroundTruthIfGiven(args, quiet);
        const std::size_t k = args.size("k", 10);
        std::vector<std::vector<std::uint32_t>> predictions;
        predictions.reserve(queries.count);
        for (std::size_t i = 0; i < queries.count; ++i) {
            predictions.push_back(
                toIds(index.searchKnn(queries.at(i), k, params.efSearch)));
        }
        std::printf("Recall@%zu at ef=%zu: %.4f\n", k, params.efSearch,
                    recallAtK(predictions, truth, k));
    }
    return 0;
}

int commandLoad(const Args& args) {
    args.validate(kCommonOptions);
    const bool quiet = args.flag("quiet");

    const Timer timer;
    HnswIndex index = HnswIndex::load(args.required("index"));
    std::printf("Loaded index from %s in %.2f s\n", args.str("index").c_str(),
                timer.elapsedSeconds());
    printStats(index);

    if (args.flag("verify") && args.has("query") && args.has("gt")) {
        const VectorSet queries = loadQueries(args, index.metric(), quiet);
        const GroundTruth truth = loadGroundTruthIfGiven(args, quiet);
        const std::size_t k = args.size("k", 10);
        const std::size_t ef = args.size("ef", index.params().efSearch);
        std::vector<std::vector<std::uint32_t>> predictions;
        predictions.reserve(queries.count);
        for (std::size_t i = 0; i < queries.count; ++i) {
            predictions.push_back(toIds(index.searchKnn(queries.at(i), k, ef)));
        }
        std::printf("Recall@%zu at ef=%zu: %.4f\n", k, ef,
                    recallAtK(predictions, truth, k));
    }
    return 0;
}

int commandSearch(const Args& args) {
    args.validate(kCommonOptions);
    const bool quiet = args.flag("quiet");

    HnswIndex index = obtainIndex(args, quiet);
    const VectorSet queries = loadQueries(args, index.metric(), quiet);
    checkNormalization(queries, index.metric(), "query");
    const GroundTruth truth = loadGroundTruthIfGiven(args, quiet);

    const std::size_t k = args.size("k", 10);
    const std::size_t ef = args.size("ef", index.params().efSearch);

    std::vector<std::vector<std::uint32_t>> predictions;
    predictions.reserve(queries.count);
    std::vector<double> latencies;
    latencies.reserve(queries.count);

    HnswIndex::SearchContext context;
    const Timer wall;
    for (std::size_t i = 0; i < queries.count; ++i) {
        const Timer single;
        const std::vector<SearchResult> found =
            index.searchKnn(queries.at(i), k, ef, context);
        latencies.push_back(single.elapsedMillis());
        predictions.push_back(toIds(found));
    }
    const double seconds = wall.elapsedSeconds();

    std::printf("Searched %u queries (k=%zu, ef=%zu) in %.3f s\n", queries.count,
                k, ef, seconds);
    std::printf("  median %.3f ms   p95 %.3f ms   %.0f qps (single thread)\n",
                median(latencies), percentile(latencies, 95.0),
                static_cast<double>(queries.count) / seconds);
    if (!truth.empty()) {
        std::printf("  Recall@%zu: %.4f\n", k, recallAtK(predictions, truth, k));
    }

    const std::size_t show = std::min<std::size_t>(args.size("show", 5), queries.count);
    for (std::size_t i = 0; i < show; ++i) {
        std::printf("  query %zu:", i);
        const std::vector<SearchResult> found =
            index.searchKnn(queries.at(i), k, ef, context);
        for (const SearchResult& r : found) {
            std::printf(" %u(%.4f)", r.id, r.distance);
        }
        std::printf("\n");
    }
    return 0;
}

int commandBenchmark(const Args& args) {
    args.validate(kCommonOptions);
    const bool quiet = args.flag("quiet");

    HnswIndex index = obtainIndex(args, quiet);
    const VectorSet queries = loadQueries(args, index.metric(), quiet);
    checkNormalization(queries, index.metric(), "query");
    const GroundTruth truth = loadGroundTruthIfGiven(args, quiet);

    BenchmarkConfig config;
    config.k = args.size("k", 10);
    config.maxQueries = args.size("queries", 0);
    config.verbose = !quiet;
    if (args.has("ef-list")) {
        config.efValues = parseSizeList(args.str("ef-list"));
    }

    if (truth.empty()) {
        std::fprintf(stderr,
                     "warning: no --gt given, so accuracy cannot be measured; "
                     "the recall column will read 0.0000\n");
    }
    std::printf("Benchmarking %u queries, k=%zu\n",
                config.maxQueries == 0
                    ? queries.count
                    : static_cast<std::uint32_t>(
                          std::min<std::size_t>(config.maxQueries, queries.count)),
                config.k);
    const std::vector<BenchmarkRow> rows =
        runBenchmark(index, queries, truth, config);
    printBenchmarkTable(rows);

    if (args.flag("brute-force") && args.has("base")) {
        const VectorSet base = loadBase(args, index.metric(), quiet);
        const BruteForceIndex exact(base, index.metric());
        // The exact scan is orders of magnitude slower, so it runs on a capped
        // number of queries unless the user asked for a specific count.
        const std::size_t exactQueries =
            config.maxQueries == 0 ? 1000 : config.maxQueries;
        const BenchmarkRow row =
            benchmarkBruteForce(exact, queries, truth, config.k, exactQueries);
        std::printf("Brute force over %zu queries: recall %.4f  median %.3f ms  "
                    "p95 %.3f ms  %.1f qps\n",
                    std::min<std::size_t>(exactQueries, queries.count), row.recall,
                    row.medianMs, row.p95Ms, row.qps);
        for (const BenchmarkRow& hnswRow : rows) {
            if (hnswRow.medianMs <= 0.0) continue;
            std::printf("  ef=%-5zu recall %.4f  speedup over brute force: %6.1fx\n",
                        hnswRow.ef, hnswRow.recall, row.medianMs / hnswRow.medianMs);
        }
    }

    if (args.has("csv")) {
        writeCsv(args.str("csv"), rows);
        std::printf("Wrote %s\n", args.str("csv").c_str());
    }
    return 0;
}

int commandBruteForce(const Args& args) {
    args.validate(kCommonOptions);
    const bool quiet = args.flag("quiet");
    const Metric metric = metricFromString(args.str("metric", "cosine"));

    const VectorSet base = loadBase(args, metric, quiet);
    const VectorSet queries = loadQueries(args, metric, quiet);
    const std::size_t k = args.size("k", 100);

    const BruteForceIndex exact(base, metric);
    const Timer timer;
    const GroundTruth truth =
        exact.computeGroundTruth(queries, k, args.size("threads", 0));
    std::printf("Exact search of %u queries against %u vectors in %.2f s\n",
                queries.count, base.count, timer.elapsedSeconds());

    if (args.has("gt")) {
        const GroundTruth reference = loadIbin(args.str("gt"));
        std::vector<std::vector<std::uint32_t>> predictions;
        predictions.reserve(truth.numQueries);
        for (std::size_t i = 0; i < truth.numQueries; ++i) {
            std::vector<std::uint32_t> row;
            row.reserve(truth.k);
            for (std::size_t j = 0; j < truth.k; ++j) {
                row.push_back(static_cast<std::uint32_t>(truth.at(i)[j]));
            }
            predictions.push_back(std::move(row));
        }
        const std::size_t compareK = std::min<std::size_t>(10, reference.k);
        std::printf("Agreement with %s at k=%zu: %.4f\n", args.str("gt").c_str(),
                    compareK, recallAtK(predictions, reference, compareK));
    }

    if (args.has("out")) {
        saveIbin(args.str("out"), truth);
        std::printf("Wrote ground truth to %s (%u x %u)\n", args.str("out").c_str(),
                    truth.numQueries, truth.k);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    const std::string command = argv[1];
    if (command == "-h" || command == "--help" || command == "help") {
        printUsage();
        return 0;
    }

    try {
        const Args args(argc, argv, 2);
        if (command == "build") return commandBuild(args, /*requireOutput=*/false);
        if (command == "save") return commandBuild(args, /*requireOutput=*/true);
        if (command == "load") return commandLoad(args);
        if (command == "search") return commandSearch(args);
        if (command == "benchmark") return commandBenchmark(args);
        if (command == "bruteforce") return commandBruteForce(args);

        std::fprintf(stderr, "error: unknown command '%s'\n\n", command.c_str());
        printUsage();
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
