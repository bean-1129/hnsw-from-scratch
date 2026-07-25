// Unit tests for the distance kernels and normalization.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "distance.hpp"

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

void testDotProduct() {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, -5.0f, 6.0f};
    // 4 - 10 + 18 = 12
    CHECK_NEAR(hnsw::dotProduct(a.data(), b.data(), 3), 12.0f, 1e-6f);

    const std::vector<float> zeros(3, 0.0f);
    CHECK_NEAR(hnsw::dotProduct(a.data(), zeros.data(), 3), 0.0f, 1e-6f);
}

void testCosineDistance() {
    std::vector<float> a{3.0f, 4.0f};       // norm 5
    std::vector<float> b{-4.0f, 3.0f};      // norm 5, orthogonal to a
    std::vector<float> c{6.0f, 8.0f};       // parallel to a
    std::vector<float> d{-3.0f, -4.0f};     // opposite to a

    hnsw::normalize(a.data(), 2);
    hnsw::normalize(b.data(), 2);
    hnsw::normalize(c.data(), 2);
    hnsw::normalize(d.data(), 2);

    CHECK_NEAR(hnsw::cosineDistance(a.data(), a.data(), 2), 0.0f, 1e-6f);
    CHECK_NEAR(hnsw::cosineDistance(a.data(), c.data(), 2), 0.0f, 1e-6f);
    CHECK_NEAR(hnsw::cosineDistance(a.data(), b.data(), 2), 1.0f, 1e-6f);
    CHECK_NEAR(hnsw::cosineDistance(a.data(), d.data(), 2), 2.0f, 1e-6f);
}

void testNormalization() {
    std::vector<float> v{3.0f, 4.0f};
    CHECK_NEAR(hnsw::l2Norm(v.data(), 2), 5.0f, 1e-6f);
    hnsw::normalize(v.data(), 2);
    CHECK_NEAR(hnsw::l2Norm(v.data(), 2), 1.0f, 1e-6f);
    CHECK_NEAR(v[0], 0.6f, 1e-6f);
    CHECK_NEAR(v[1], 0.8f, 1e-6f);

    // A zero vector has no direction and must be left alone (no NaNs).
    std::vector<float> zero{0.0f, 0.0f};
    hnsw::normalize(zero.data(), 2);
    CHECK(!std::isnan(zero[0]) && !std::isnan(zero[1]));
}

void testSquaredL2() {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 6.0f, 3.0f};
    // 9 + 16 + 0 = 25
    CHECK_NEAR(hnsw::squaredL2Distance(a.data(), b.data(), 3), 25.0f, 1e-5f);
    CHECK_NEAR(hnsw::squaredL2Distance(a.data(), a.data(), 3), 0.0f, 1e-6f);
}

void testMetricDispatch() {
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{0.0f, 1.0f};
    CHECK_NEAR(hnsw::distance(a.data(), b.data(), 2, hnsw::Metric::Cosine), 1.0f,
               1e-6f);
    CHECK_NEAR(hnsw::distance(a.data(), b.data(), 2, hnsw::Metric::L2), 2.0f,
               1e-6f);
    CHECK(hnsw::metricFromString("cosine") == hnsw::Metric::Cosine);
    CHECK(hnsw::metricFromString("L2") == hnsw::Metric::L2);
}

/// For unit vectors, `1 - dot` and `0.5 * ||a - b||^2` are the same ranking
/// function; this is why normalizing once at load time is enough.
void testCosineMatchesL2OnUnitVectors() {
    std::mt19937 rng(12345);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    const std::size_t dim = 32;

    for (int trial = 0; trial < 100; ++trial) {
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = gaussian(rng);
            b[i] = gaussian(rng);
        }
        hnsw::normalize(a.data(), dim);
        hnsw::normalize(b.data(), dim);

        const float cosine = hnsw::cosineDistance(a.data(), b.data(), dim);
        const float half = 0.5f * hnsw::squaredL2Distance(a.data(), b.data(), dim);
        CHECK_NEAR(cosine, half, 1e-4f);
    }
}

}  // namespace

int main() {
    testDotProduct();
    testCosineDistance();
    testNormalization();
    testSquaredL2();
    testMetricDispatch();
    testCosineMatchesL2OnUnitVectors();

    if (g_failures == 0) {
        std::printf("test_distance: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("test_distance: %d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
}
