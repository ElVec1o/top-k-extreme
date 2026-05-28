// radix_topk.hpp
//
// Radix-select Top-K for uint64 keys.
//
// Mechanism (per 8-bit radix pass):
//   1. Histogram input by one byte (256 counters, 1 KB, L1-resident).
//   2. Walk counters from highest bucket down, accumulating, until the
//      running sum reaches K. The bucket where we cross is the "pivot
//      bucket" b*. Items with that byte STRICTLY GREATER than b* are
//      definitely in Top-K. Items EQUAL to b* are candidates; items
//      LESS THAN b* are not.
//   3. Emit pass: write definite winners to the output, and items in the
//      pivot bucket to a scratch array.
//   4. Either recurse on the scratch with the next byte (we don't need
//      to here — the scratch is small, std::nth_element finishes it) or
//      run nth_element on the scratch to pick the remaining K' items.
//
// Why this is fast at large K:
//   * Hot path is a single shift + array index + increment. Histogram
//     fits in L1, so all 5M reads stay at DRAM bandwidth on the LOAD
//     side and L1 on the STORE side. ~3-4 instructions per item.
//   * Emit pass: one compare + branchless store. Bandwidth-bound.
//   * Final nth_element runs on ~N/256 items at most — trivially fast.
//
// Three independent literature sources converged on this primitive:
//   - Floyd-Rivest 1975 (sample-based)
//   - Alabi/Blanchard/Gordon/Steinbach 2012 (radix-select for GPU)
//   - RadiK (Li et al. ICS 2024)
// All three apply the same insight: partition by high bits, recurse on
// the pivot partition.

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace algo4 {

// 8-way independent histogram on the top byte of each uint64.
// Standard trick to expose more instruction-level parallelism to the
// CPU and avoid serialising the read-modify-write dependency chain
// when consecutive items map to the same counter. Each h*[256] is 1 KB,
// so 8 of them = 8 KB, still L1-resident on M4 (192 KB L1d per P-core).
static inline void histogram_top_byte_8way(const uint64_t* data, size_t n,
                                           uint32_t out[256]) {
    uint32_t h0[256] = {}, h1[256] = {}, h2[256] = {}, h3[256] = {};
    uint32_t h4[256] = {}, h5[256] = {}, h6[256] = {}, h7[256] = {};
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        ++h0[data[i + 0] >> 56];
        ++h1[data[i + 1] >> 56];
        ++h2[data[i + 2] >> 56];
        ++h3[data[i + 3] >> 56];
        ++h4[data[i + 4] >> 56];
        ++h5[data[i + 5] >> 56];
        ++h6[data[i + 6] >> 56];
        ++h7[data[i + 7] >> 56];
    }
    for (; i < n; ++i) ++h0[data[i] >> 56];
    for (int b = 0; b < 256; ++b) {
        out[b] = h0[b] + h1[b] + h2[b] + h3[b]
               + h4[b] + h5[b] + h6[b] + h7[b];
    }
}

// Single-pass radix-select.
//
// Returns the K largest values (unsorted). Caller sorts if order matters.
//
// Algorithm:
//   * Histogram by top byte.
//   * Walk down from bucket 255 to find pivot b* where cumulative count
//     first meets or exceeds K.
//   * Items with top-byte > b* are emitted directly to the output.
//   * Items with top-byte == b* are written to scratch.
//   * If we have more than K total candidates, nth_element on scratch
//     to pick exactly K_remaining of them.
inline std::vector<uint64_t> radix_select_topk(const std::vector<uint64_t>& data,
                                               size_t K) {
    const size_t N = data.size();
    if (K >= N) return data;
    if (K == 0) return {};

    // 1. Histogram.
    uint32_t counts[256];
    histogram_top_byte_8way(data.data(), N, counts);

    // 2. Find pivot bucket.
    size_t cumsum = 0;
    int pivot = 0;
    for (int b = 255; b >= 0; --b) {
        cumsum += counts[b];
        if (cumsum >= K) { pivot = b; break; }
    }
    const size_t above_pivot = cumsum - counts[pivot];        // strictly > pivot
    const size_t take_from_pivot = K - above_pivot;           // need this many from pivot bucket

    // 3. Emit pass. We pre-reserve based on exact counts from the
    // histogram so push_back never reallocates. The branchful variant
    // (preserved here) ended up matching or beating a fully-branchless
    // version on M4 because the doubled write traffic of unconditional
    // stores outweighed the saved branch on a workload where the
    // 3-way branch is well-predicted in the steady state.
    std::vector<uint64_t> out;
    out.reserve(above_pivot + 16);
    std::vector<uint64_t> scratch;
    scratch.reserve(counts[pivot] + 16);

    const uint64_t pivot_byte = (uint64_t)pivot;
    for (size_t i = 0; i < N; ++i) {
        uint64_t v = data[i];
        uint64_t top = v >> 56;
        if (top > pivot_byte)        out.push_back(v);
        else if (top == pivot_byte)  scratch.push_back(v);
    }

    // 4. Resolve the pivot bucket.
    if (take_from_pivot > 0 && take_from_pivot < scratch.size()) {
        // Pick the top `take_from_pivot` items in scratch.
        std::nth_element(scratch.begin(),
                         scratch.begin() + take_from_pivot - 1,
                         scratch.end(),
                         std::greater<uint64_t>());
        scratch.resize(take_from_pivot);
    } else if (take_from_pivot >= scratch.size()) {
        // Take all of scratch (rare edge case: pivot bucket exactly meets K).
    } else {
        scratch.clear();
    }

    out.insert(out.end(), scratch.begin(), scratch.end());
    // out now has exactly K items.
    return out;
}

} // namespace algo4
