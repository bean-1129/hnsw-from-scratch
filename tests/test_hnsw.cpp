// Unit tests for the HNSW index: structural invariants of the graph, search
// correctness against the brute force oracle, determinism and serialization.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "brute_force.hpp"
#include "hnsw.hpp"
#include "recall.hpp"

namespace {

int g_failures = 0;

#define CHECK(condition)                                                    \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++g_failures;                                                   \
        }                                                                   \
    } while (false)

#define CHECK_NEAR(a, b, tolerance) CHECK(std::fabs((a) - (b)) <= (tolerance))

std::string tempPath(const char* name) {
    const char* dir = std::getenv("TMPDIR");
    std::string base = dir != nullptr ? dir : "/tmp";
    if (!base.empty() && base.back() != '/') base.push_back('/');
    return base + "hnsw_test_" + name;
}

hnsw::VectorSet randomVectors(std::uint32_t count, std::uint32_t dim,
                              std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);

    hnsw::VectorSet vectors;
    vectors.count = count;
    vectors.dim = dim;
    vectors.data.resize(static_cast<std::size_t>(count) * dim);
    for (float& value : vectors.data) value = gaussian(rng);
    vectors.normalizeAll();
    return vectors;
}

/// Clustered data exercises the neighbor selection heuristic much harder than
/// uniform noise: without diversity pruning the graph fragments into cliques.
///
/// The spread is deliberately wide (0.3 per coordinate against unit-variance
/// centers). Tighter clusters turn every cluster into a set of near duplicates
/// whose true top-10 is decided by differences of ~1e-3 in distance; recall is
/// then dominated by tie breaking rather than by graph quality, which makes for
/// a fixture that punishes a correct implementation.
hnsw::VectorSet clusteredVectors(std::uint32_t clusters, std::uint32_t perCluster,
                                 std::uint32_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::normal_distribution<float> jitter(0.0f, 0.3f);

    hnsw::VectorSet vectors;
    vectors.count = clusters * perCluster;
    vectors.dim = dim;
    vectors.data.resize(static_cast<std::size_t>(vectors.count) * dim);

    std::vector<float> center(dim);
    for (std::uint32_t c = 0; c < clusters; ++c) {
        for (std::uint32_t d = 0; d < dim; ++d) center[d] = gaussian(rng);
        for (std::uint32_t i = 0; i < perCluster; ++i) {
            float* row = vectors.at(static_cast<std::size_t>(c) * perCluster + i);
            for (std::uint32_t d = 0; d < dim; ++d) {
                row[d] = center[d] + jitter(rng);
            }
        }
    }
    vectors.normalizeAll();
    return vectors;
}

hnsw::HnswParams defaultParams() {
    hnsw::HnswParams params;
    params.M = 16;
    params.efConstruction = 200;
    params.efSearch = 64;
    params.seed = 42;
    return params;
}

double measureRecall(const hnsw::HnswIndex& index,
                     const hnsw::BruteForceIndex& exact,
                     const hnsw::VectorSet& queries, std::size_t k,
                     std::size_t ef) {
    const hnsw::GroundTruth truth = exact.computeGroundTruth(queries, k, 1);
    std::vector<std::vector<std::uint32_t>> predicted;
    predicted.reserve(queries.count);
    for (std::size_t i = 0; i < queries.count; ++i) {
        predicted.push_back(hnsw::toIds(index.searchKnn(queries.at(i), k, ef)));
    }
    return hnsw::recallAtK(predicted, truth, k);
}

// -- tests ------------------------------------------------------------------

void testEmptyAndSingleElement() {
    hnsw::HnswIndex index(4, defaultParams());
    CHECK(index.size() == 0);
    CHECK(index.entryPoint() == hnsw::kInvalidId);

    const std::vector<float> query{1.0f, 0.0f, 0.0f, 0.0f};
    CHECK(index.searchKnn(query.data(), 10).empty());

    const std::vector<float> point{0.0f, 1.0f, 0.0f, 0.0f};
    const std::uint32_t id = index.addPoint(point.data());
    CHECK(id == 0);
    CHECK(index.size() == 1);
    CHECK(index.entryPoint() == 0);

    const std::vector<hnsw::SearchResult> found = index.searchKnn(query.data(), 10);
    CHECK(found.size() == 1);
    CHECK(found[0].id == 0);
    CHECK_NEAR(found[0].distance, 1.0f, 1e-6f);
}

/// Every point must be able to find itself with distance 0.
void testFindsItself() {
    const hnsw::VectorSet vectors = randomVectors(1000, 16, 5);
    hnsw::HnswIndex index(vectors.dim, defaultParams());
    index.build(vectors);
    CHECK(index.size() == vectors.count);

    std::size_t misses = 0;
    for (std::size_t i = 0; i < vectors.count; ++i) {
        const std::vector<hnsw::SearchResult> found =
            index.searchKnn(vectors.at(i), 1, 64);
        CHECK(!found.empty());
        if (found.empty() || found[0].id != i) ++misses;
    }
    // Self retrieval is the easiest possible query; a correct implementation
    // should essentially never miss it.
    CHECK(misses <= 2);
}

void testGraphInvariants() {
    const hnsw::VectorSet vectors = randomVectors(2000, 24, 6);
    hnsw::HnswParams params = defaultParams();
    params.M = 8;
    hnsw::HnswIndex index(vectors.dim, params);
    index.build(vectors);

    const hnsw::Graph& graph = index.graph();
    CHECK(graph.size() == vectors.count);
    CHECK(graph.maxM() == 8);
    CHECK(graph.maxM0() == 16);
    CHECK(index.maxLevel() >= 0);
    CHECK(graph.topLayer() == index.maxLevel());
    CHECK(graph.levelOf(index.entryPoint()) == index.maxLevel());

    std::size_t isolated = 0;
    for (std::uint32_t id = 0; id < graph.size(); ++id) {
        const int level = graph.levelOf(id);
        CHECK(level >= 0);
        for (int layer = 0; layer <= level; ++layer) {
            const std::vector<std::uint32_t>& neighbors = graph.neighbors(id, layer);
            // Degree cap: Mmax0 on layer 0, Mmax above.
            CHECK(neighbors.size() <= graph.maxDegree(layer));
            for (std::uint32_t neighbor : neighbors) {
                CHECK(neighbor != id);                    // no self loops
                CHECK(neighbor < graph.size());           // valid id
                CHECK(graph.levelOf(neighbor) >= layer);  // exists on this layer
            }
            // No duplicate edges.
            std::vector<std::uint32_t> sorted = neighbors;
            std::sort(sorted.begin(), sorted.end());
            CHECK(std::unique(sorted.begin(), sorted.end()) == sorted.end());
        }
        if (graph.neighbors(id, 0).empty()) ++isolated;
    }
    CHECK(isolated == 0);

    // The level distribution must decay geometrically with ratio ~1/M.
    CHECK(graph.nodesOnLayer(0) == vectors.count);
    for (int layer = 1; layer <= graph.topLayer(); ++layer) {
        CHECK(graph.nodesOnLayer(layer) <= graph.nodesOnLayer(layer - 1));
    }
    CHECK(graph.nodesOnLayer(1) < vectors.count / 2);
}

/// The exponential level distribution: P(level >= l) ~ (1/M)^l.
void testLevelDistribution() {
    const hnsw::VectorSet vectors = randomVectors(20000, 4, 8);
    hnsw::HnswParams params = defaultParams();
    params.M = 16;
    params.efConstruction = 16;  // keep the test fast; levels are unaffected
    hnsw::HnswIndex index(vectors.dim, params);
    index.build(vectors);

    const hnsw::Graph& graph = index.graph();
    const double atLayer1 = static_cast<double>(graph.nodesOnLayer(1)) /
                            static_cast<double>(graph.size());
    // Expected fraction is exp(-1) ^ ... = 1/M = 0.0625 for mL = 1/ln(M).
    CHECK(atLayer1 > 0.03 && atLayer1 < 0.11);
}

void testRecallAgainstBruteForce() {
    const hnsw::VectorSet base = randomVectors(5000, 32, 11);
    const hnsw::VectorSet queries = randomVectors(200, 32, 12);

    hnsw::HnswIndex index(base.dim, defaultParams());
    index.build(base);
    const hnsw::BruteForceIndex exact(base);

    // 32-dimensional isotropic Gaussians are close to the worst case for any
    // graph index (no low intrinsic dimensionality to exploit), so the ef=32
    // bar is set where a correct implementation lands with margin, while ef=256
    // must be essentially exact.
    const double recallLow = measureRecall(index, exact, queries, 10, 32);
    const double recallHigh = measureRecall(index, exact, queries, 10, 256);
    std::printf("  recall@10: ef=32 -> %.4f, ef=256 -> %.4f\n", recallLow,
                recallHigh);

    CHECK(recallLow > 0.85);
    CHECK(recallHigh > 0.99);
    // Recall must be monotone in ef (up to noise).
    CHECK(recallHigh >= recallLow - 1e-9);
}

void testRecallOnClusteredData() {
    const hnsw::VectorSet base = clusteredVectors(50, 100, 32, 13);
    const hnsw::VectorSet queries = clusteredVectors(100, 1, 32, 14);

    hnsw::HnswIndex index(base.dim, defaultParams());
    index.build(base);
    const hnsw::BruteForceIndex exact(base);

    const double moderate = measureRecall(index, exact, queries, 10, 128);
    const double wide = measureRecall(index, exact, queries, 10, 512);
    std::printf("  clustered recall@10: ef=128 -> %.4f, ef=512 -> %.4f\n",
                moderate, wide);
    CHECK(moderate > 0.90);
    CHECK(wide > 0.99);
}

void testResultsAreSorted() {
    const hnsw::VectorSet base = randomVectors(2000, 16, 15);
    hnsw::HnswIndex index(base.dim, defaultParams());
    index.build(base);

    const hnsw::VectorSet queries = randomVectors(50, 16, 16);
    for (std::size_t q = 0; q < queries.count; ++q) {
        const std::vector<hnsw::SearchResult> found =
            index.searchKnn(queries.at(q), 10, 64);
        CHECK(found.size() == 10);
        for (std::size_t i = 1; i < found.size(); ++i) {
            CHECK(found[i - 1].distance <= found[i].distance);
            CHECK(found[i - 1].id != found[i].id);
        }
        // Reported distances must match a direct recomputation.
        for (const hnsw::SearchResult& r : found) {
            const float expected =
                hnsw::cosineDistance(queries.at(q), index.vectorAt(r.id), base.dim);
            CHECK_NEAR(r.distance, expected, 1e-5f);
        }
    }
}

/// ef is clamped up to k, so a small ef never truncates the result list.
void testEfSmallerThanK() {
    const hnsw::VectorSet base = randomVectors(500, 8, 17);
    hnsw::HnswIndex index(base.dim, defaultParams());
    index.build(base);

    const std::vector<hnsw::SearchResult> found =
        index.searchKnn(base.at(0), 20, 1);
    CHECK(found.size() == 20);
}

void testDeterminism() {
    const hnsw::VectorSet base = randomVectors(1500, 16, 18);
    const hnsw::VectorSet queries = randomVectors(30, 16, 19);

    hnsw::HnswIndex first(base.dim, defaultParams());
    first.build(base);
    hnsw::HnswIndex second(base.dim, defaultParams());
    second.build(base);

    CHECK(first.maxLevel() == second.maxLevel());
    CHECK(first.entryPoint() == second.entryPoint());
    for (std::size_t q = 0; q < queries.count; ++q) {
        const std::vector<hnsw::SearchResult> a = first.searchKnn(queries.at(q), 10, 64);
        const std::vector<hnsw::SearchResult> b = second.searchKnn(queries.at(q), 10, 64);
        CHECK(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].id == b[i].id);
    }

    // A different seed must produce a different level assignment.
    hnsw::HnswParams other = defaultParams();
    other.seed = 4242;
    hnsw::HnswIndex third(base.dim, other);
    third.build(base);
    CHECK(third.graph().nodesOnLayer(1) != first.graph().nodesOnLayer(1) ||
          third.entryPoint() != first.entryPoint());
}

void testSerializationRoundTrip() {
    const hnsw::VectorSet base = randomVectors(1200, 16, 20);
    const hnsw::VectorSet queries = randomVectors(40, 16, 21);

    hnsw::HnswParams params = defaultParams();
    params.M = 12;
    params.efConstruction = 120;
    params.efSearch = 77;

    hnsw::HnswIndex original(base.dim, params);
    original.build(base);

    const std::string path = tempPath("index.hnsw");
    original.save(path);

    const hnsw::HnswIndex restored = hnsw::HnswIndex::load(path);
    CHECK(restored.size() == original.size());
    CHECK(restored.dim() == original.dim());
    CHECK(restored.maxLevel() == original.maxLevel());
    CHECK(restored.entryPoint() == original.entryPoint());
    CHECK(restored.params().M == 12);
    CHECK(restored.params().efConstruction == 120);
    CHECK(restored.params().efSearch == 77);
    CHECK(restored.params().seed == params.seed);
    CHECK(restored.metric() == params.metric);
    CHECK(restored.graph().totalEdges() == original.graph().totalEdges());

    for (std::uint32_t id = 0; id < original.size(); ++id) {
        CHECK(restored.graph().levelOf(id) == original.graph().levelOf(id));
        for (int layer = 0; layer <= original.graph().levelOf(id); ++layer) {
            CHECK(restored.graph().neighbors(id, layer) ==
                  original.graph().neighbors(id, layer));
        }
    }
    for (std::size_t q = 0; q < queries.count; ++q) {
        const std::vector<hnsw::SearchResult> a = original.searchKnn(queries.at(q), 10, 64);
        const std::vector<hnsw::SearchResult> b = restored.searchKnn(queries.at(q), 10, 64);
        CHECK(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].id == b[i].id);
            CHECK_NEAR(a[i].distance, b[i].distance, 1e-6f);
        }
    }
    std::remove(path.c_str());
}

void testLoadRejectsGarbage() {
    const std::string path = tempPath("garbage.hnsw");
    {
        std::FILE* file = std::fopen(path.c_str(), "wb");
        CHECK(file != nullptr);
        const char junk[32] = {0};
        std::fwrite(junk, 1, sizeof(junk), file);
        std::fclose(file);
    }
    bool threw = false;
    try {
        hnsw::HnswIndex::load(path);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(path.c_str());
}

void testL2Index() {
    hnsw::VectorSet base;
    base.count = 2000;
    base.dim = 8;
    base.data.resize(static_cast<std::size_t>(base.count) * base.dim);
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    for (float& value : base.data) value = uniform(rng);

    hnsw::HnswParams params = defaultParams();
    params.metric = hnsw::Metric::L2;
    hnsw::HnswIndex index(base.dim, params);
    index.build(base);

    hnsw::VectorSet queries;
    queries.count = 50;
    queries.dim = base.dim;
    queries.data.resize(static_cast<std::size_t>(queries.count) * queries.dim);
    for (float& value : queries.data) value = uniform(rng);

    const hnsw::BruteForceIndex exact(base, hnsw::Metric::L2);
    const double recall = measureRecall(index, exact, queries, 10, 128);
    std::printf("  L2 recall@10 at ef=128: %.4f\n", recall);
    CHECK(recall > 0.95);
}

/// keepPrunedConnections=false must still yield a usable (if slightly worse)
/// graph, and extendCandidates must not break anything.
void testHeuristicVariants() {
    const hnsw::VectorSet base = clusteredVectors(40, 50, 16, 24);
    const hnsw::VectorSet queries = randomVectors(50, 16, 25);
    const hnsw::BruteForceIndex exact(base);

    hnsw::HnswParams pruned = defaultParams();
    pruned.keepPrunedConnections = false;
    hnsw::HnswIndex withoutPruned(base.dim, pruned);
    withoutPruned.build(base);
    const double recallA = measureRecall(withoutPruned, exact, queries, 10, 128);

    hnsw::HnswParams extended = defaultParams();
    extended.extendCandidates = true;
    hnsw::HnswIndex withExtended(base.dim, extended);
    withExtended.build(base);
    const double recallB = measureRecall(withExtended, exact, queries, 10, 128);

    std::printf("  keepPruned=false -> %.4f, extendCandidates=true -> %.4f\n",
                recallA, recallB);
    CHECK(recallA > 0.90);
    CHECK(recallB > 0.90);

    // Without keepPrunedConnections the graph is sparser by construction.
    hnsw::HnswIndex baseline(base.dim, defaultParams());
    baseline.build(base);
    CHECK(withoutPruned.graph().totalEdges() <= baseline.graph().totalEdges());
}

void testIncrementalInsertion() {
    const hnsw::VectorSet base = randomVectors(800, 12, 26);
    hnsw::HnswIndex index(base.dim, defaultParams());
    for (std::size_t i = 0; i < base.count; ++i) {
        const std::uint32_t id = index.addPoint(base.at(i));
        CHECK(id == i);
        // The index must be queryable after every single insertion.
        if (i % 100 == 0) {
            const std::vector<hnsw::SearchResult> found =
                index.searchKnn(base.at(0), 1, 32);
            CHECK(!found.empty());
            CHECK(found[0].id == 0);
        }
    }
    CHECK(index.size() == base.count);
}

void testStats() {
    const hnsw::VectorSet base = randomVectors(1000, 16, 27);
    hnsw::HnswIndex index(base.dim, defaultParams());
    index.build(base);

    const hnsw::IndexStats stats = index.stats();
    CHECK(stats.numElements == 1000);
    CHECK(stats.dim == 16);
    CHECK(stats.nodesPerLayer.size() == static_cast<std::size_t>(stats.maxLevel) + 1);
    CHECK(stats.nodesPerLayer[0] == 1000);
    CHECK(stats.totalEdges > 0);
    CHECK(stats.averageDegreePerLayer[0] > 1.0);
    CHECK(stats.vectorBytes >= 1000 * 16 * sizeof(float));
}

void testDimensionMismatchThrows() {
    const hnsw::VectorSet base = randomVectors(10, 8, 28);
    hnsw::HnswIndex index(16, defaultParams());
    bool threw = false;
    try {
        index.build(base);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    testEmptyAndSingleElement();
    testFindsItself();
    testGraphInvariants();
    testLevelDistribution();
    testRecallAgainstBruteForce();
    testRecallOnClusteredData();
    testResultsAreSorted();
    testEfSmallerThanK();
    testDeterminism();
    testSerializationRoundTrip();
    testLoadRejectsGarbage();
    testL2Index();
    testHeuristicVariants();
    testIncrementalInsertion();
    testStats();
    testDimensionMismatchThrows();

    if (g_failures == 0) {
        std::printf("test_hnsw: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("test_hnsw: %d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
}
