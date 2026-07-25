// distance.hpp -- vector metrics.
//
// The project follows the "normalize once, at load time" rule: every base and
// query vector is scaled to unit length by the loader, after which
//
//     cosine_similarity(a, b) = dot(a, b)
//     cosine_distance(a, b)   = 1 - dot(a, b)
//
// so no norms are ever computed inside the search loops.
//
// The kernels are deliberately plain scalar loops (no intrinsics). At -O3 the
// compiler auto-vectorizes them; hand written SIMD is discussed in the README
// as the next optimization step.
#pragma once

#include <cstddef>
#include <string>

namespace hnsw {

/// Metrics supported by the index. `Cosine` assumes unit-length vectors.
enum class Metric {
    Cosine,  ///< 1 - dot(a, b), valid for normalized vectors.
    L2       ///< squared Euclidean distance (monotone in the true L2 distance).
};

/// Inner product of two `dim`-dimensional vectors.
float dotProduct(const float* a, const float* b, std::size_t dim);

/// Cosine distance for *pre-normalized* vectors: 1 - dot(a, b).
/// The result lies in [0, 2]; smaller means more similar.
float cosineDistance(const float* a, const float* b, std::size_t dim);

/// Squared Euclidean distance. Monotone in the Euclidean distance, so it can be
/// used directly for ranking without the square root.
float squaredL2Distance(const float* a, const float* b, std::size_t dim);

/// Dispatches to the kernel selected by `metric`.
float distance(const float* a, const float* b, std::size_t dim, Metric metric);

/// Euclidean norm of a vector.
float l2Norm(const float* v, std::size_t dim);

/// Scales `v` to unit length in place. Zero vectors are left untouched.
void normalize(float* v, std::size_t dim);

const char* metricName(Metric metric);

/// Parses "cosine" / "l2"; throws std::invalid_argument on anything else.
Metric metricFromString(const std::string& name);

}  // namespace hnsw
