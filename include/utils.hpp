// utils.hpp -- small shared utilities: timing, statistics, binary I/O helpers,
// a constant-time "visited set" and string parsing for the command line.
//
// Nothing in here is specific to HNSW; it is the plumbing used by every other
// module of the project.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnsw {

/// Sentinel used wherever a node id may be absent (e.g. an empty index has no
/// entry point).
constexpr std::uint32_t kInvalidId = 0xFFFFFFFFu;

/// A single (id, distance) pair returned by any of the search routines.
/// `distance` is always a *distance*: smaller means closer.
struct SearchResult {
    std::uint32_t id = kInvalidId;
    float distance = 0.0f;
};

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Monotonic stopwatch. Construction starts the clock.
class Timer {
public:
    Timer() : start_(Clock::now()) {}

    void reset() { start_ = Clock::now(); }

    double elapsedSeconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    double elapsedMillis() const { return elapsedSeconds() * 1000.0; }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Percentile of an unsorted sample using linear interpolation between the two
/// nearest ranks. `p` is expressed in percent, i.e. 50.0 is the median.
double percentile(std::vector<double> values, double p);

/// Convenience wrapper for `percentile(values, 50.0)`.
double median(std::vector<double> values);

/// Arithmetic mean; returns 0 for an empty sample.
double mean(const std::vector<double>& values);

// ---------------------------------------------------------------------------
// Visited set
// ---------------------------------------------------------------------------

/// Epoch based membership test used by the graph traversals.
///
/// A naive implementation would allocate an `unordered_set` per query, which
/// dominates the runtime of a beam search. Instead we keep one byte-cheap tag
/// per node and bump a generation counter between queries, so clearing the set
/// is O(1) instead of O(n).
class VisitedList {
public:
    VisitedList() = default;

    /// Ensures the list can hold `n` node ids. Cheap when already large enough.
    void resize(std::size_t n) {
        if (marks_.size() < n) marks_.resize(n, 0);
    }

    /// Starts a new epoch: every element is considered unvisited again.
    void reset() {
        if (++epoch_ == 0) {  // wrapped around: clear the tags for real
            std::fill(marks_.begin(), marks_.end(), 0u);
            epoch_ = 1;
        }
    }

    bool visited(std::uint32_t id) const { return marks_[id] == epoch_; }

    /// Marks `id` as visited and returns true if it was *not* visited before.
    bool markVisited(std::uint32_t id) {
        if (marks_[id] == epoch_) return false;
        marks_[id] = epoch_;
        return true;
    }

private:
    std::vector<std::uint32_t> marks_;
    std::uint32_t epoch_ = 0;
};

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------

/// Writes a trivially copyable value in native byte order.
template <typename T>
void writePod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("utils: failed to write to stream");
}

/// Reads a trivially copyable value in native byte order.
template <typename T>
void readPod(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("utils: unexpected end of stream");
}

/// Writes `count` elements from a contiguous buffer.
template <typename T>
void writeArray(std::ostream& out, const T* data, std::size_t count) {
    if (count == 0) return;
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) throw std::runtime_error("utils: failed to write to stream");
}

/// Reads `count` elements into a contiguous buffer.
template <typename T>
void readArray(std::istream& in, T* data, std::size_t count) {
    if (count == 0) return;
    in.read(reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) throw std::runtime_error("utils: unexpected end of stream");
}

// ---------------------------------------------------------------------------
// Strings / filesystem
// ---------------------------------------------------------------------------

std::vector<std::string> splitString(const std::string& text, char delimiter);

/// Parses a comma separated list of non-negative integers ("16,32,64").
std::vector<std::size_t> parseSizeList(const std::string& text);

/// Human readable byte count, e.g. "1.42 GiB".
std::string formatBytes(std::size_t bytes);

bool fileExists(const std::string& path);

/// Creates every missing directory component of `path`'s parent directory.
/// Silently does nothing when the path has no directory component.
void ensureParentDirectory(const std::string& path);

}  // namespace hnsw
