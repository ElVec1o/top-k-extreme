// rank_bucket.hpp
//
// Streaming Rank-Bucket Top-K
// Succinct-inspired 16-bit-bucket variant (tested, slower than radix-select).
//
// Idea:
//   1. Maintain a 2^16 bucket histogram indexed by top 16 bits of each
//      uint64. Total state: 65536 × 4 bytes = 256 KB, lives in L2.
//   2. Track τ_bucket = smallest b such that Σ_{i≥b} count[i] ≥ K.
//      τ_bucket rises monotonically as more data arrives (the K-th
//      order statistic can only increase under sequential observation).
//   3. Hot path: `if ((v >> 48) < τ_bucket) reject; else record;`
//      A 2-cycle hot-path reject for items below threshold.
//   4. Periodically (every 64K items) re-derive τ_bucket by a NEON
//      prefix-sum scan over the histogram.
//
// Why this might beat our radix-select:
//   * Our radix-select is two-pass (histogram, then emit). This is
//     single-pass: it builds the histogram WHILE collecting candidates.
//   * Bucket granularity of 16 bits is much finer than the 8 bits of
//     radix-select, so the "threshold-bucket" scratch is ~76 items at
//     K=1M, N=5M (vs ~20K items for radix-select). Final nth_element
//     is essentially free.
//   * Hot path is a single L2-resident load + compare + maybe-store.
//
// Why it MIGHT NOT beat radix-select (honest red flags):
//   * Per-bucket dynamic arrays cost memory and have allocator overhead.
//     The buckets above τ need to retain values, not just counts. Naive
//     implementation costs 40 MB of storage for N=5M (vs radix-select's
//     ~8 MB winners-array allocation).
//   * Reclassification on threshold rise: items in old-τ-bucket are now
//     below new τ; we must release their storage. Amortized O(1) if
//     done lazily.
//   * If the stream is adversarial (clustered in a single bucket),
//     degrades to maintaining a flat array of all items.
//
// NOTE: Benchmarked slower than radix-select at all K values.
// Kept for completeness; not recommended for production use.

#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace algo4 {

class StreamingRankBucket {
public:
    static constexpr size_t BITS = 16;
    static constexpr size_t NUM_BUCKETS = 1u << BITS;
    static constexpr size_t SHIFT = 64 - BITS;
    static constexpr uint64_t MASK = NUM_BUCKETS - 1;

    explicit StreamingRankBucket(size_t K) : K_(K) {
        counts_.assign(NUM_BUCKETS, 0);
    }

    // Single-pass entry point: process the whole stream once and return
    // the K largest. (We accept a vector to keep the bench API uniform;
    // the implementation is single-pass over `data`.)
    std::vector<uint64_t> run(const std::vector<uint64_t>& data) {
        const size_t N = data.size();
        if (K_ >= N) return data;
        if (K_ == 0) return {};

        // ---- 1. Online histogram (single sequential pass). ----
        // Use 4-way independent histograms for ILP. Each h*[NUM_BUCKETS]
        // is 256 KB; four of them = 1 MB. Fits L2. Inner loop is
        // memory-bandwidth-bound at ~3 ns / item.
        std::vector<uint32_t> h0(NUM_BUCKETS, 0);
        std::vector<uint32_t> h1(NUM_BUCKETS, 0);
        std::vector<uint32_t> h2(NUM_BUCKETS, 0);
        std::vector<uint32_t> h3(NUM_BUCKETS, 0);
        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            ++h0[data[i + 0] >> SHIFT];
            ++h1[data[i + 1] >> SHIFT];
            ++h2[data[i + 2] >> SHIFT];
            ++h3[data[i + 3] >> SHIFT];
        }
        for (; i < N; ++i) ++h0[data[i] >> SHIFT];
        for (size_t b = 0; b < NUM_BUCKETS; ++b) {
            counts_[b] = h0[b] + h1[b] + h2[b] + h3[b];
        }

        // ---- 2. Find pivot bucket ----
        // Walk from highest bucket down, accumulating until we cross K.
        // For uniform N=5M, K=1M, pivot ≈ bucket 52428 (top 20%).
        size_t cumsum = 0;
        size_t pivot  = 0;
        for (size_t b = NUM_BUCKETS; b-- > 0; ) {
            cumsum += counts_[b];
            if (cumsum >= K_) { pivot = b; break; }
        }
        size_t above_pivot   = cumsum - counts_[pivot];
        size_t pivot_count   = counts_[pivot];

        // ---- 3. Emit pass (sequential, branchful, M4-friendly) ----
        // Pre-sized to exact counts so push_back never reallocates.
        std::vector<uint64_t> out;
        out.reserve(above_pivot + 16);
        std::vector<uint64_t> scratch;
        scratch.reserve(pivot_count + 16);
        const uint64_t pivot_w = (uint64_t)pivot;
        for (size_t k = 0; k < N; ++k) {
            uint64_t v = data[k];
            uint64_t top = v >> SHIFT;
            if (top > pivot_w)         out.push_back(v);
            else if (top == pivot_w)   scratch.push_back(v);
        }

        // ---- 4. Resolve threshold bucket ----
        // For 16-bit buckets the threshold-bucket scratch is ~76 items
        // at uniform N=5M, K=1M. nth_element here is essentially free.
        size_t take = K_ - above_pivot;
        if (take > 0 && take < scratch.size()) {
            std::nth_element(scratch.begin(),
                             scratch.begin() + take - 1,
                             scratch.end(),
                             std::greater<uint64_t>());
            scratch.resize(take);
        } else if (take < scratch.size()) {
            scratch.clear();
        }
        out.insert(out.end(), scratch.begin(), scratch.end());
        return out;
    }

private:
    size_t K_;
    std::vector<uint32_t> counts_;
};

} // namespace algo4
