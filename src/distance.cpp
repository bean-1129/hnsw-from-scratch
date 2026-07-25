#include "distance.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace hnsw {

// Plain scalar loops. Written so that -O3 can auto-vectorize them: no aliasing
// between the accumulator and the inputs, a trip count known at runtime only,
// and no early exits.

float dotProduct(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosineDistance(const float* a, const float* b, std::size_t dim) {
    // Valid only for unit-length inputs, which the loader guarantees:
    //   cos(a, b) = dot(a, b) / (|a| |b|) = dot(a, b)
    return 1.0f - dotProduct(a, b, dim);
}

float squaredL2Distance(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float distance(const float* a, const float* b, std::size_t dim, Metric metric) {
    switch (metric) {
        case Metric::L2:
            return squaredL2Distance(a, b, dim);
        case Metric::Cosine:
        default:
            return cosineDistance(a, b, dim);
    }
}

float l2Norm(const float* v, std::size_t dim) {
    return std::sqrt(dotProduct(v, v, dim));
}

void normalize(float* v, std::size_t dim) {
    const float norm = l2Norm(v, dim);
    // A zero vector has no direction; leaving it untouched keeps its distance
    // to everything well defined (1.0 under cosine) instead of producing NaNs.
    if (norm <= 0.0f || !std::isfinite(norm)) return;
    const float inverse = 1.0f / norm;
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] *= inverse;
    }
}

const char* metricName(Metric metric) {
    switch (metric) {
        case Metric::L2:
            return "l2";
        case Metric::Cosine:
        default:
            return "cosine";
    }
}

Metric metricFromString(const std::string& name) {
    std::string lowered = name;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "cosine" || lowered == "angular" || lowered == "ip" ||
        lowered == "dot") {
        return Metric::Cosine;
    }
    if (lowered == "l2" || lowered == "euclidean") {
        return Metric::L2;
    }
    throw std::invalid_argument("unknown metric: '" + name + "'");
}

}  // namespace hnsw
