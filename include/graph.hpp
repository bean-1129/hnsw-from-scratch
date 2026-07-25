// graph.hpp -- the layered proximity graph behind the HNSW index.
//
// Responsibilities are split deliberately:
//
//   * `Graph` owns the *topology*  : per node level + adjacency list per layer.
//   * `HnswIndex` owns the *geometry*: the contiguous vector storage and every
//     algorithm from the paper.
//
// Keeping the two apart makes the algorithms in hnsw.cpp read like the
// pseudocode of the paper, and lets the tests inspect the graph directly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace hnsw {

/// Layered adjacency structure.
///
/// A node inserted at level `l` exists on layers `0 … l`; layer 0 contains
/// every node. Adjacency lists are capped at `maxM0` on layer 0 and `maxM` on
/// every layer above (Mmax0 / Mmax in the paper).
class Graph {
public:
    using NeighborList = std::vector<std::uint32_t>;

    Graph() = default;
    Graph(std::size_t maxM, std::size_t maxM0);

    /// Appends a node living on layers `0 … level` and returns its id.
    std::uint32_t addNode(int level);

    int levelOf(std::uint32_t id) const { return nodes_[id].level; }

    const NeighborList& neighbors(std::uint32_t id, int layer) const {
        return nodes_[id].links[static_cast<std::size_t>(layer)];
    }
    NeighborList& neighbors(std::uint32_t id, int layer) {
        return nodes_[id].links[static_cast<std::size_t>(layer)];
    }

    void setNeighbors(std::uint32_t id, int layer, NeighborList list);

    /// Appends `neighbor` to the adjacency list of `id` on `layer` if there is
    /// room left. Returns false when the list is already at its cap, in which
    /// case the caller has to re-run the neighbor selection heuristic.
    bool tryAddNeighbor(std::uint32_t id, int layer, std::uint32_t neighbor);

    /// Cap on the number of neighbors a node may keep on `layer`.
    std::size_t maxDegree(int layer) const { return layer == 0 ? maxM0_ : maxM_; }

    std::size_t size() const { return nodes_.size(); }
    bool empty() const { return nodes_.empty(); }
    std::size_t maxM() const { return maxM_; }
    std::size_t maxM0() const { return maxM0_; }

    /// Highest layer index that currently holds at least one node, or -1 when
    /// the graph is empty.
    int topLayer() const { return topLayer_; }

    // -- statistics ---------------------------------------------------------

    std::size_t nodesOnLayer(int layer) const;
    double averageDegree(int layer) const;
    std::size_t totalEdges() const;
    /// Bytes held by the adjacency lists (capacity, not just size).
    std::size_t memoryUsageBytes() const;

    // -- serialization ------------------------------------------------------

    void save(std::ostream& out) const;
    void load(std::istream& in);

private:
    struct Node {
        int level = 0;
        /// links[l] holds the neighbors on layer l, for l in [0, level].
        std::vector<NeighborList> links;
    };

    std::vector<Node> nodes_;
    std::size_t maxM_ = 16;
    std::size_t maxM0_ = 32;
    int topLayer_ = -1;
};

}  // namespace hnsw
