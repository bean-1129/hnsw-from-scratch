#include "vector_loader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "utils.hpp"

namespace hnsw {
namespace {

/// Guards against absurd headers (a truncated or non-.fbin file would
/// otherwise trigger a multi-terabyte allocation).
void validateHeader(const std::string& path, std::uint32_t count,
                    std::uint32_t dim, std::size_t bytesPerElement,
                    std::size_t fileSize) {
    if (dim == 0) {
        throw std::runtime_error("loader: '" + path + "' declares dim = 0");
    }
    const std::size_t expected =
        8 + static_cast<std::size_t>(count) * dim * bytesPerElement;
    if (fileSize < expected) {
        throw std::runtime_error("loader: '" + path + "' is truncated (header announces " +
                                 std::to_string(expected) + " bytes, file has " +
                                 std::to_string(fileSize) + ")");
    }
}

std::size_t fileSizeOf(std::ifstream& stream) {
    const std::streampos current = stream.tellg();
    stream.seekg(0, std::ios::end);
    const std::streampos end = stream.tellg();
    stream.seekg(current, std::ios::beg);
    return static_cast<std::size_t>(end);
}

}  // namespace

void VectorSet::normalizeAll() {
    for (std::size_t i = 0; i < count; ++i) {
        normalize(at(i), dim);
    }
}

float VectorSet::maxNormDeviation() const {
    float worst = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float norm = l2Norm(at(i), dim);
        worst = std::max(worst, std::fabs(norm - 1.0f));
    }
    return worst;
}

VectorSet VectorSet::head(std::size_t n) const {
    VectorSet subset;
    subset.dim = dim;
    subset.count = static_cast<std::uint32_t>(std::min<std::size_t>(n, count));
    const std::size_t floats = static_cast<std::size_t>(subset.count) * dim;
    subset.data.assign(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(floats));
    return subset;
}

VectorSet loadFbin(const std::string& path, std::size_t maxVectors) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("loader: cannot open '" + path + "'");
    }

    const std::size_t bytes = fileSizeOf(in);
    if (bytes < 8) {
        throw std::runtime_error("loader: '" + path + "' is too small to be a .fbin file");
    }

    VectorSet vectors;
    readPod(in, vectors.count);
    readPod(in, vectors.dim);
    validateHeader(path, vectors.count, vectors.dim, sizeof(float), bytes);

    if (maxVectors > 0 && maxVectors < vectors.count) {
        vectors.count = static_cast<std::uint32_t>(maxVectors);
    }

    const std::size_t floats =
        static_cast<std::size_t>(vectors.count) * vectors.dim;
    vectors.data.resize(floats);
    readArray(in, vectors.data.data(), floats);
    return vectors;
}

GroundTruth loadIbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("loader: cannot open '" + path + "'");
    }

    const std::size_t bytes = fileSizeOf(in);
    if (bytes < 8) {
        throw std::runtime_error("loader: '" + path + "' is too small to be an .ibin file");
    }

    GroundTruth truth;
    readPod(in, truth.numQueries);
    readPod(in, truth.k);
    validateHeader(path, truth.numQueries, truth.k, sizeof(std::int32_t), bytes);

    const std::size_t total =
        static_cast<std::size_t>(truth.numQueries) * truth.k;
    truth.neighbors.resize(total);
    readArray(in, truth.neighbors.data(), total);
    return truth;
}

void saveFbin(const std::string& path, const VectorSet& vectors) {
    ensureParentDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("loader: cannot write '" + path + "'");
    }
    writePod(out, vectors.count);
    writePod(out, vectors.dim);
    writeArray(out, vectors.data.data(), vectors.data.size());
}

void saveIbin(const std::string& path, const GroundTruth& groundTruth) {
    ensureParentDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("loader: cannot write '" + path + "'");
    }
    writePod(out, groundTruth.numQueries);
    writePod(out, groundTruth.k);
    writeArray(out, groundTruth.neighbors.data(), groundTruth.neighbors.size());
}

VectorSet loadVectors(const std::string& path, Metric metric,
                      std::size_t maxVectors) {
    VectorSet vectors = loadFbin(path, maxVectors);
    // Normalization happens exactly once, here. Everything downstream -- graph
    // construction, search, brute force -- then treats cosine distance as
    // `1 - dot` and never computes a norm again.
    if (metric == Metric::Cosine) {
        vectors.normalizeAll();
    }
    return vectors;
}

}  // namespace hnsw
