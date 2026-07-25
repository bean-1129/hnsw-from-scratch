// vector_loader.hpp -- readers/writers for the project's binary formats.
//
// The C++ side never touches HDF5. `python/convert_hdf5.py` turns the
// ann-benchmarks HDF5 archive into two flat files:
//
//   *.fbin   uint32 count | uint32 dim | float32 count*dim  (row major)
//   *.ibin   uint32 count | uint32 k   | int32   count*k    (row major)
//
// Both are written in the host's native byte order, which is little endian on
// every platform this project targets.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "distance.hpp"

namespace hnsw {

/// A dense, row-major matrix of float vectors kept in one contiguous buffer.
///
/// Contiguous storage matters: the HNSW graph walk is bound by random memory
/// access, and one allocation per dataset keeps rows a fixed stride apart.
struct VectorSet {
    std::uint32_t count = 0;  ///< number of vectors
    std::uint32_t dim = 0;    ///< dimensionality of each vector
    std::vector<float> data;  ///< count * dim floats, row major

    /// Pointer to the i-th vector. No bounds checking in release builds.
    const float* at(std::size_t i) const {
        return data.data() + i * static_cast<std::size_t>(dim);
    }
    float* at(std::size_t i) {
        return data.data() + i * static_cast<std::size_t>(dim);
    }

    std::size_t size() const { return count; }
    bool empty() const { return count == 0; }

    /// Scales every row to unit length. Called exactly once, right after
    /// loading, so that cosine distance degenerates to `1 - dot`.
    void normalizeAll();

    /// Largest absolute deviation of any row norm from 1.0. Used by the CLI to
    /// warn when a cosine index is fed un-normalized data.
    float maxNormDeviation() const;

    /// Copy of the first `n` rows (`n >= count` returns a full copy).
    VectorSet head(std::size_t n) const;
};

/// Ground truth neighbor ids: `numQueries` rows of `k` ids into the base set.
struct GroundTruth {
    std::uint32_t numQueries = 0;
    std::uint32_t k = 0;
    std::vector<std::int32_t> neighbors;

    const std::int32_t* at(std::size_t i) const {
        return neighbors.data() + i * static_cast<std::size_t>(k);
    }

    bool empty() const { return numQueries == 0; }
};

/// Loads a `.fbin` file. `maxVectors == 0` means "read everything"; otherwise
/// only the first `maxVectors` rows are read (the rest of the file is skipped
/// without being touched).
VectorSet loadFbin(const std::string& path, std::size_t maxVectors = 0);

/// Loads a `.ibin` ground truth file.
GroundTruth loadIbin(const std::string& path);

void saveFbin(const std::string& path, const VectorSet& vectors);
void saveIbin(const std::string& path, const GroundTruth& groundTruth);

/// Loads a `.fbin` and, when `metric == Metric::Cosine`, normalizes it once.
/// This is the entry point used by every command of the CLI.
VectorSet loadVectors(const std::string& path, Metric metric,
                      std::size_t maxVectors = 0);

}  // namespace hnsw
