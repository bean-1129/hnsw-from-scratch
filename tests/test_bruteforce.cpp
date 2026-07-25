// Unit tests for the exact search oracle.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "brute_force.hpp"

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

hnsw::VectorSet randomVectors(std::uint32_t count, std::uint32_t dim,
                              std::uint32_t seed, bool normalized) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);

    hnsw::VectorSet vectors;
    vectors.count = count;
    vectors.dim = dim;
    vectors.data.resize(static_cast<std::size_t>(count) * dim);
    for (float& value : vectors.data) value = gaussian(rng);
    if (normalized) vectors.normalizeAll();
    return vectors;
}

void testKnownLayout() {
    // Four points on the unit circle; the nearest neighbor of (1, 0) is itself,
    // then the two orthogonal points, then the antipode.
    hnsw::VectorSet vectors;
    vectors.count = 4;
    vectors.dim = 2;
    vectors.data = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f};

    const hnsw::BruteForceIndex index(vectors, hnsw::Metric::Cosine);
    const std::vector<float> query{1.0f, 0.0f};
    const std::vector<hnsw::SearchResult> found = index.search(query.data(), 4);

    CHECK(found.size() == 4);
    CHECK(found[0].id == 0);
    CHECK_NEAR(found[0].distance, 0.0f, 1e-6f);
    CHECK(found[3].id == 2);  // the antipode is farthest
    CHECK_NEAR(found[3].distance, 2.0f, 1e-6f);
    // The two orthogonal points tie at distance 1.
    CHECK_NEAR(found[1].distance, 1.0f, 1e-6f);
    CHECK_NEAR(found[2].distance, 1.0f, 1e-6f);
}

void testResultsAreSortedAndUnique() {
    const hnsw::VectorSet vectors = randomVectors(500, 16, 7, true);
    const hnsw::BruteForceIndex index(vectors);
    const hnsw::VectorSet queries = randomVectors(20, 16, 99, true);

    for (std::size_t q = 0; q < queries.count; ++q) {
        const std::vector<hnsw::SearchResult> found =
            index.search(queries.at(q), 10);
        CHECK(found.size() == 10);
        for (std::size_t i = 1; i < found.size(); ++i) {
            CHECK(found[i - 1].distance <= found[i].distance);
            CHECK(found[i - 1].id != found[i].id);
        }
    }
}

/// The heap based top-k must agree with a full sort of all distances.
void testMatchesFullSort() {
    const hnsw::VectorSet vectors = randomVectors(300, 8, 3, true);
    const hnsw::BruteForceIndex index(vectors);
    const hnsw::VectorSet queries = randomVectors(10, 8, 4, true);
    const std::size_t k = 15;

    for (std::size_t q = 0; q < queries.count; ++q) {
        std::vector<hnsw::SearchResult> all;
        all.reserve(vectors.count);
        for (std::size_t i = 0; i < vectors.count; ++i) {
            all.push_back({static_cast<std::uint32_t>(i),
                           hnsw::cosineDistance(queries.at(q), vectors.at(i),
                                                vectors.dim)});
        }
        std::sort(all.begin(), all.end(),
                  [](const hnsw::SearchResult& a, const hnsw::SearchResult& b) {
                      return a.distance < b.distance;
                  });

        const std::vector<hnsw::SearchResult> found = index.search(queries.at(q), k);
        for (std::size_t i = 0; i < k; ++i) {
            CHECK_NEAR(found[i].distance, all[i].distance, 1e-6f);
        }
    }
}

void testKLargerThanDataset() {
    const hnsw::VectorSet vectors = randomVectors(5, 4, 11, true);
    const hnsw::BruteForceIndex index(vectors);
    const std::vector<hnsw::SearchResult> found = index.search(vectors.at(0), 50);
    CHECK(found.size() == 5);
    CHECK(found[0].id == 0);
}

void testBatchMatchesSingle() {
    const hnsw::VectorSet vectors = randomVectors(400, 12, 21, true);
    const hnsw::BruteForceIndex index(vectors);
    const hnsw::VectorSet queries = randomVectors(64, 12, 22, true);

    const std::vector<std::vector<std::uint32_t>> batch =
        index.searchBatch(queries, 10, 4);
    CHECK(batch.size() == queries.count);
    for (std::size_t q = 0; q < queries.count; ++q) {
        const std::vector<hnsw::SearchResult> single = index.search(queries.at(q), 10);
        CHECK(batch[q].size() == single.size());
        for (std::size_t i = 0; i < single.size(); ++i) {
            CHECK(batch[q][i] == single[i].id);
        }
    }
}

void testGroundTruthPacking() {
    const hnsw::VectorSet vectors = randomVectors(200, 6, 31, true);
    const hnsw::BruteForceIndex index(vectors);
    const hnsw::VectorSet queries = randomVectors(15, 6, 32, true);

    const hnsw::GroundTruth truth = index.computeGroundTruth(queries, 10, 2);
    CHECK(truth.numQueries == 15);
    CHECK(truth.k == 10);
    CHECK(truth.neighbors.size() == 150);
    for (std::size_t q = 0; q < truth.numQueries; ++q) {
        const std::vector<hnsw::SearchResult> single = index.search(queries.at(q), 10);
        CHECK(truth.at(q)[0] == static_cast<std::int32_t>(single[0].id));
    }
}

void testL2Metric() {
    hnsw::VectorSet vectors;
    vectors.count = 3;
    vectors.dim = 2;
    vectors.data = {0.0f, 0.0f, 10.0f, 0.0f, 1.0f, 1.0f};

    const hnsw::BruteForceIndex index(vectors, hnsw::Metric::L2);
    const std::vector<float> query{0.5f, 0.5f};
    const std::vector<hnsw::SearchResult> found = index.search(query.data(), 3);
    CHECK(found[0].id == 2);  // (1,1) is closest to (0.5,0.5)
    CHECK(found[2].id == 1);  // (10,0) is farthest
}

}  // namespace

int main() {
    testKnownLayout();
    testResultsAreSortedAndUnique();
    testMatchesFullSort();
    testKLargerThanDataset();
    testBatchMatchesSingle();
    testGroundTruthPacking();
    testL2Metric();

    if (g_failures == 0) {
        std::printf("test_bruteforce: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("test_bruteforce: %d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
}
