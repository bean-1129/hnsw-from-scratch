// Unit tests for the Recall@k evaluator.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

hnsw::GroundTruth makeGroundTruth(std::uint32_t numQueries, std::uint32_t k,
                                  std::vector<std::int32_t> ids) {
    hnsw::GroundTruth truth;
    truth.numQueries = numQueries;
    truth.k = k;
    truth.neighbors = std::move(ids);
    return truth;
}

void testPerfectRecall() {
    const hnsw::GroundTruth truth = makeGroundTruth(2, 3, {1, 2, 3, 4, 5, 6});
    const std::vector<std::vector<std::uint32_t>> predicted{{1, 2, 3}, {4, 5, 6}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 3), 1.0, 1e-9);
}

void testZeroRecall() {
    const hnsw::GroundTruth truth = makeGroundTruth(1, 3, {1, 2, 3});
    const std::vector<std::vector<std::uint32_t>> predicted{{7, 8, 9}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 3), 0.0, 1e-9);
}

/// Recall is set based: order inside the top-k does not matter.
void testOrderInsensitive() {
    const hnsw::GroundTruth truth = makeGroundTruth(1, 4, {1, 2, 3, 4});
    const std::vector<std::vector<std::uint32_t>> predicted{{4, 3, 2, 1}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 4), 1.0, 1e-9);
}

void testPartialRecall() {
    // Two of four ids match -> 0.5 for that query; the second query matches
    // one of four -> 0.25. Mean = 0.375.
    const hnsw::GroundTruth truth =
        makeGroundTruth(2, 4, {1, 2, 3, 4, 10, 11, 12, 13});
    const std::vector<std::vector<std::uint32_t>> predicted{{1, 2, 99, 98},
                                                            {10, 50, 51, 52}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 4), 0.375, 1e-9);
}

/// Evaluating at k = 10 when the ground truth stores 100 neighbors must only
/// look at the first 10 ground truth ids.
void testEvaluatesPrefixOfGroundTruth() {
    std::vector<std::int32_t> ids;
    for (std::int32_t i = 0; i < 100; ++i) ids.push_back(i);
    const hnsw::GroundTruth truth = makeGroundTruth(1, 100, std::move(ids));

    const std::vector<std::vector<std::uint32_t>> predicted{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 10), 1.0, 1e-9);

    // Ids 50..59 are correct neighbors but not in the true top-10.
    const std::vector<std::vector<std::uint32_t>> late{
        {50, 51, 52, 53, 54, 55, 56, 57, 58, 59}};
    CHECK_NEAR(hnsw::recallAtK(late, truth, 10), 0.0, 1e-9);
}

void testSubsetOfQueries() {
    const hnsw::GroundTruth truth = makeGroundTruth(3, 2, {1, 2, 3, 4, 5, 6});
    const std::vector<std::vector<std::uint32_t>> predicted{{1, 2}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 2), 1.0, 1e-9);
}

void testShortResultRow() {
    // An index that returned fewer than k neighbors is penalized, not crashing.
    const hnsw::GroundTruth truth = makeGroundTruth(1, 4, {1, 2, 3, 4});
    const std::vector<std::vector<std::uint32_t>> predicted{{1, 2}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 4), 0.5, 1e-9);
}

void testRejectsTooLargeK() {
    const hnsw::GroundTruth truth = makeGroundTruth(1, 2, {1, 2});
    const std::vector<std::vector<std::uint32_t>> predicted{{1, 2}};
    bool threw = false;
    try {
        hnsw::recallAtK(predicted, truth, 10);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void testSearchResultOverload() {
    const hnsw::GroundTruth truth = makeGroundTruth(1, 3, {5, 6, 7});
    const std::vector<std::vector<hnsw::SearchResult>> predicted{
        {{5, 0.1f}, {6, 0.2f}, {99, 0.3f}}};
    CHECK_NEAR(hnsw::recallAtK(predicted, truth, 3), 2.0 / 3.0, 1e-9);

    const std::vector<std::uint32_t> ids = hnsw::toIds(predicted[0]);
    CHECK(ids.size() == 3);
    CHECK(ids[0] == 5 && ids[2] == 99);
}

}  // namespace

int main() {
    testPerfectRecall();
    testZeroRecall();
    testOrderInsensitive();
    testPartialRecall();
    testEvaluatesPrefixOfGroundTruth();
    testSubsetOfQueries();
    testShortResultRow();
    testRejectsTooLargeK();
    testSearchResultOverload();

    if (g_failures == 0) {
        std::printf("test_recall: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("test_recall: %d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
}
