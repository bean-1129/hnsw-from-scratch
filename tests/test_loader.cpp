// Unit tests for the binary loaders, including the normalize-once contract.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "utils.hpp"
#include "vector_loader.hpp"

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

hnsw::VectorSet makeVectors() {
    hnsw::VectorSet vectors;
    vectors.count = 3;
    vectors.dim = 2;
    vectors.data = {3.0f, 4.0f,    // norm 5
                    0.0f, 2.0f,    // norm 2
                    -1.0f, 0.0f};  // norm 1
    return vectors;
}

void testFbinRoundTrip() {
    const std::string path = tempPath("roundtrip.fbin");
    const hnsw::VectorSet original = makeVectors();
    hnsw::saveFbin(path, original);

    const hnsw::VectorSet loaded = hnsw::loadFbin(path);
    CHECK(loaded.count == original.count);
    CHECK(loaded.dim == original.dim);
    CHECK(loaded.data == original.data);
    CHECK_NEAR(loaded.at(1)[1], 2.0f, 1e-6f);
    std::remove(path.c_str());
}

void testFbinHeaderLayout() {
    // The on-disk header must literally be `uint32 count | uint32 dim`.
    const std::string path = tempPath("header.fbin");
    hnsw::saveFbin(path, makeVectors());

    std::ifstream in(path, std::ios::binary);
    std::uint32_t count = 0, dim = 0;
    hnsw::readPod(in, count);
    hnsw::readPod(in, dim);
    CHECK(count == 3);
    CHECK(dim == 2);

    float first = 0.0f;
    hnsw::readPod(in, first);
    CHECK_NEAR(first, 3.0f, 1e-6f);
    in.close();
    std::remove(path.c_str());
}

void testPartialLoad() {
    const std::string path = tempPath("partial.fbin");
    hnsw::saveFbin(path, makeVectors());

    const hnsw::VectorSet head = hnsw::loadFbin(path, 2);
    CHECK(head.count == 2);
    CHECK(head.dim == 2);
    CHECK(head.data.size() == 4);
    std::remove(path.c_str());
}

void testNormalizationOnLoad() {
    const std::string path = tempPath("normalize.fbin");
    hnsw::saveFbin(path, makeVectors());

    const hnsw::VectorSet raw = hnsw::loadVectors(path, hnsw::Metric::L2);
    CHECK_NEAR(raw.at(0)[0], 3.0f, 1e-6f);  // L2 must NOT normalize

    const hnsw::VectorSet unit = hnsw::loadVectors(path, hnsw::Metric::Cosine);
    for (std::size_t i = 0; i < unit.count; ++i) {
        CHECK_NEAR(hnsw::l2Norm(unit.at(i), unit.dim), 1.0f, 1e-6f);
    }
    CHECK_NEAR(unit.at(0)[0], 0.6f, 1e-6f);
    CHECK_NEAR(unit.at(0)[1], 0.8f, 1e-6f);
    CHECK(unit.maxNormDeviation() < 1e-6f);
    std::remove(path.c_str());
}

void testIbinRoundTrip() {
    const std::string path = tempPath("groundtruth.ibin");
    hnsw::GroundTruth truth;
    truth.numQueries = 2;
    truth.k = 3;
    truth.neighbors = {7, 1, 4, 0, 9, 2};
    hnsw::saveIbin(path, truth);

    const hnsw::GroundTruth loaded = hnsw::loadIbin(path);
    CHECK(loaded.numQueries == 2);
    CHECK(loaded.k == 3);
    CHECK(loaded.at(0)[0] == 7);
    CHECK(loaded.at(1)[2] == 2);
    std::remove(path.c_str());
}

void testMissingFileThrows() {
    bool threw = false;
    try {
        hnsw::loadFbin(tempPath("definitely_missing.fbin"));
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void testTruncatedFileThrows() {
    // Header claims 100 vectors, body holds none.
    const std::string path = tempPath("truncated.fbin");
    {
        std::ofstream out(path, std::ios::binary);
        const std::uint32_t count = 100, dim = 8;
        hnsw::writePod(out, count);
        hnsw::writePod(out, dim);
    }
    bool threw = false;
    try {
        hnsw::loadFbin(path);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(path.c_str());
}

void testHead() {
    const hnsw::VectorSet vectors = makeVectors();
    const hnsw::VectorSet head = vectors.head(2);
    CHECK(head.count == 2);
    CHECK(head.data.size() == 4);
    CHECK(vectors.head(99).count == vectors.count);
}

}  // namespace

int main() {
    testFbinRoundTrip();
    testFbinHeaderLayout();
    testPartialLoad();
    testNormalizationOnLoad();
    testIbinRoundTrip();
    testMissingFileThrows();
    testTruncatedFileThrows();
    testHead();

    if (g_failures == 0) {
        std::printf("test_loader: all checks passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("test_loader: %d check(s) failed\n", g_failures);
    return EXIT_FAILURE;
}
