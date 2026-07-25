#include "utils.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace hnsw {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    if (values.size() == 1) return values.front();

    const double clamped = std::min(100.0, std::max(0.0, p));
    // Linear interpolation between closest ranks (the "R-7" definition, which
    // is what numpy.percentile uses by default).
    const double rank = clamped / 100.0 * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    const double weight = rank - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double median(std::vector<double> values) {
    return percentile(std::move(values), 50.0);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

std::vector<std::string> splitString(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string item;
    std::istringstream stream(text);
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty()) parts.push_back(item);
    }
    return parts;
}

std::vector<std::size_t> parseSizeList(const std::string& text) {
    std::vector<std::size_t> values;
    for (const std::string& part : splitString(text, ',')) {
        try {
            values.push_back(static_cast<std::size_t>(std::stoull(part)));
        } catch (const std::exception&) {
            throw std::invalid_argument("not an integer list: '" + text + "'");
        }
    }
    if (values.empty()) {
        throw std::invalid_argument("empty integer list: '" + text + "'");
    }
    return values;
}

std::string formatBytes(std::size_t bytes) {
    static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f %s", value, kUnits[unit]);
    return std::string(buffer);
}

bool fileExists(const std::string& path) {
    struct stat info;
    return ::stat(path.c_str(), &info) == 0;
}

void ensureParentDirectory(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;

    const std::string parent = path.substr(0, slash);
    // Create every missing component, ignoring "already exists" errors.
    std::string current;
    for (std::size_t i = 0; i <= parent.size(); ++i) {
        if (i == parent.size() || parent[i] == '/') {
            if (!current.empty() && !fileExists(current)) {
                ::mkdir(current.c_str(), 0775);
            }
        }
        if (i < parent.size()) current.push_back(parent[i]);
    }
}

}  // namespace hnsw
