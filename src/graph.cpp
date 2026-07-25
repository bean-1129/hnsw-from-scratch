#include "graph.hpp"

#include <algorithm>
#include <stdexcept>

#include "utils.hpp"

namespace hnsw {
namespace {
constexpr std::uint32_t kGraphFormatVersion = 1;
}

Graph::Graph(std::size_t maxM, std::size_t maxM0)
    : maxM_(maxM), maxM0_(maxM0) {
    if (maxM_ == 0 || maxM0_ == 0) {
        throw std::invalid_argument("graph: maxM and maxM0 must be positive");
    }
}

std::uint32_t Graph::addNode(int level) {
    if (level < 0) throw std::invalid_argument("graph: negative level");

    Node node;
    node.level = level;
    node.links.resize(static_cast<std::size_t>(level) + 1);
    // Reserving the cap up front means an insertion never reallocates an
    // adjacency list, which keeps the neighborhoods compact in memory.
    for (int layer = 0; layer <= level; ++layer) {
        node.links[static_cast<std::size_t>(layer)].reserve(maxDegree(layer));
    }

    nodes_.push_back(std::move(node));
    topLayer_ = std::max(topLayer_, level);
    return static_cast<std::uint32_t>(nodes_.size() - 1);
}

void Graph::setNeighbors(std::uint32_t id, int layer, NeighborList list) {
    NeighborList& target = neighbors(id, layer);
    target = std::move(list);
}

bool Graph::tryAddNeighbor(std::uint32_t id, int layer, std::uint32_t neighbor) {
    NeighborList& list = neighbors(id, layer);
    if (list.size() >= maxDegree(layer)) return false;
    list.push_back(neighbor);
    return true;
}

std::size_t Graph::nodesOnLayer(int layer) const {
    std::size_t total = 0;
    for (const Node& node : nodes_) {
        if (node.level >= layer) ++total;
    }
    return total;
}

double Graph::averageDegree(int layer) const {
    std::size_t nodes = 0;
    std::size_t edges = 0;
    for (const Node& node : nodes_) {
        if (node.level < layer) continue;
        ++nodes;
        edges += node.links[static_cast<std::size_t>(layer)].size();
    }
    if (nodes == 0) return 0.0;
    return static_cast<double>(edges) / static_cast<double>(nodes);
}

std::size_t Graph::totalEdges() const {
    std::size_t edges = 0;
    for (const Node& node : nodes_) {
        for (const NeighborList& list : node.links) edges += list.size();
    }
    return edges;
}

std::size_t Graph::memoryUsageBytes() const {
    std::size_t bytes = nodes_.capacity() * sizeof(Node);
    for (const Node& node : nodes_) {
        bytes += node.links.capacity() * sizeof(NeighborList);
        for (const NeighborList& list : node.links) {
            bytes += list.capacity() * sizeof(std::uint32_t);
        }
    }
    return bytes;
}

void Graph::save(std::ostream& out) const {
    writePod(out, kGraphFormatVersion);
    writePod(out, static_cast<std::uint64_t>(maxM_));
    writePod(out, static_cast<std::uint64_t>(maxM0_));
    writePod(out, static_cast<std::uint64_t>(nodes_.size()));
    writePod(out, static_cast<std::int32_t>(topLayer_));

    for (const Node& node : nodes_) {
        writePod(out, static_cast<std::int32_t>(node.level));
        for (const NeighborList& list : node.links) {
            writePod(out, static_cast<std::uint32_t>(list.size()));
            writeArray(out, list.data(), list.size());
        }
    }
}

void Graph::load(std::istream& in) {
    std::uint32_t version = 0;
    readPod(in, version);
    if (version != kGraphFormatVersion) {
        throw std::runtime_error("graph: unsupported format version " +
                                 std::to_string(version));
    }

    std::uint64_t maxM = 0, maxM0 = 0, numNodes = 0;
    std::int32_t topLayer = -1;
    readPod(in, maxM);
    readPod(in, maxM0);
    readPod(in, numNodes);
    readPod(in, topLayer);

    maxM_ = static_cast<std::size_t>(maxM);
    maxM0_ = static_cast<std::size_t>(maxM0);
    topLayer_ = topLayer;

    nodes_.clear();
    nodes_.resize(static_cast<std::size_t>(numNodes));
    for (Node& node : nodes_) {
        std::int32_t level = 0;
        readPod(in, level);
        if (level < 0) throw std::runtime_error("graph: corrupt node level");
        node.level = level;
        node.links.resize(static_cast<std::size_t>(level) + 1);
        for (int layer = 0; layer <= level; ++layer) {
            NeighborList& list = node.links[static_cast<std::size_t>(layer)];
            std::uint32_t degree = 0;
            readPod(in, degree);
            if (degree > maxDegree(layer)) {
                throw std::runtime_error("graph: node degree exceeds the cap");
            }
            list.reserve(maxDegree(layer));
            list.resize(degree);
            readArray(in, list.data(), degree);
        }
    }
}

}  // namespace hnsw
