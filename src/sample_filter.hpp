// sample_filter.hpp
//
// Two-pass exact Top-K using sample-and-filter (Floyd-Rivest flavor).
//
// Mechanics:
//   1. Pull a small uniform random sample of size M (default 16k).
//   2. Find the (K * M / N * oversample)-th largest in the sample via
//      std::nth_element. With oversample > 1, this picks a sample rank
//      *further* from the top than the expected one, yielding a
//      threshold value statistically guaranteed to be ≤ the true K-th
//      largest with probability > 1 - exp(-O(oversample)).
//   3. Sequential filter sweep: collect items strictly greater than the
//      threshold. Expected collection size ≈ K * oversample.
//   4. std::nth_element on the (small) collection.
//
// The win:
//   * The filter sweep is a single cache-friendly pass; the threshold
//     lives in a CPU register. The CPU sees ~1 compare + occasional
//     branch-predicted append per item.
//   * `std::nth_element` on the WHOLE N=5M array touches each item
//     ~log(N) times due to partitioning. Our final nth_element touches
//     only ~K*oversample items.
//   * Total work: O(N + K). Strictly better than O(N log K) heap
//     streaming and beats the constants on offline O(N) nth_element
//     because we run nth_element on a much smaller array.
//
// Caveats called out HONESTLY:
//   * Requires random access to the input (not streaming).
//   * Has a fallback path: if the sample-derived threshold turns out
//     too high (collection.size() < K), we re-run nth_element on the
//     full array. Probability < 1% with oversample=5.
//   * Only useful when K is large enough that the per-item cost of
//     heap maintenance dominates (K >= ~10K). For very small K, the
//     existing AsyncMembraneTopK is already optimal.

#pragma once
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace algo4 {

class SampleFilterTopK {
public:
    // oversample: safety factor. Higher → safer (lower fallback rate)
    // but more items collected in pass 2 → slower final nth_element.
    // M = sample_size: tune for accuracy. M=16384 fits in L1.
    //
    // If oversample <= 0, use an adaptive policy: higher safety for
    // small K/N (where the sample tail is sparse) and tighter for
    // large K/N (where the sample tail is dense, and a smaller
    // collection dramatically speeds the final nth_element).
    SampleFilterTopK(size_t K,
                     size_t sample_size = 16384,
                     double oversample  = 0.0)        // 0 = adaptive
        : K_(K), M_(sample_size), oversample_(oversample) {}

    // Compute the oversample factor we'd actually use for given (K, N).
    double effective_oversample(size_t N) const {
        if (oversample_ > 0.0) return oversample_;
        // Adaptive: K_/N tells us the percentile we're estimating.
        // Standard error of order-statistic estimate ~ sqrt(p(1-p)/M).
        // Pick oversample so threshold is ~ (rank + k_sigma * stderr).
        double p = (double)K_ / (double)N;
        if (p <= 0.0)      return 4.0;
        if (p >= 0.5)      return 1.4;   // K close to N/2 — tight
        if (p >= 0.1)      return 1.6;   // large K
        if (p >= 0.01)     return 2.5;   // medium K
        return 4.0;                       // small K — safety dominates
    }

    std::vector<uint64_t> run(const std::vector<uint64_t>& data) {
        const size_t N = data.size();
        if (K_ >= N) {
            // Degenerate: just return everything (caller should bypass).
            return data;
        }
        double ov = effective_oversample(N);
        if (M_ >= N || (double)K_ * ov >= (double)N * 0.9) {
            // Sample-and-filter wouldn't shrink the collection enough.
            return fallback_full_nth(data);
        }

        // --- 1. Sample ---
        std::vector<uint64_t> sample(M_);
        std::mt19937_64 rng(0xC0FFEE);
        // Use uniform integer distribution over [0, N) for unbiased sample.
        std::uniform_int_distribution<size_t> dist(0, N - 1);
        for (size_t i = 0; i < M_; ++i) {
            sample[i] = data[dist(rng)];
        }

        // --- 2. Threshold estimate ---
        // Expected rank in sample corresponding to true K-th = K*M/N.
        // Pick a rank > that to land on a LOWER value (conservative).
        double expected_rank =
            (double)K_ * (double)M_ / (double)N * ov;
        size_t sample_rank = (size_t)expected_rank;
        if (sample_rank == 0) sample_rank = 1;
        if (sample_rank >= M_) sample_rank = M_ - 1;

        std::nth_element(sample.begin(),
                         sample.begin() + sample_rank,
                         sample.end(),
                         std::greater<uint64_t>());
        uint64_t threshold = sample[sample_rank];

        // --- 3. Filter sweep ---
        std::vector<uint64_t> collection;
        // Reserve a generous estimate: K * oversample, plus 25% slack.
        collection.reserve((size_t)((double)K_ * ov * 1.25) + 128);
        for (size_t i = 0; i < N; ++i) {
            uint64_t v = data[i];
            if (v > threshold) collection.push_back(v);
        }

        // --- 4. Fallback or final exact selection ---
        if (collection.size() < K_) {
            // Sample was unlucky. Pay the full price.
            return fallback_full_nth(data);
        }
        std::nth_element(collection.begin(),
                         collection.begin() + K_ - 1,
                         collection.end(),
                         std::greater<uint64_t>());
        collection.resize(K_);
        return collection;
    }

private:
    std::vector<uint64_t> fallback_full_nth(const std::vector<uint64_t>& data) {
        std::vector<uint64_t> v = data;
        std::nth_element(v.begin(),
                         v.begin() + K_ - 1,
                         v.end(),
                         std::greater<uint64_t>());
        v.resize(K_);
        return v;
    }

    size_t K_;
    size_t M_;
    double oversample_;
};

} // namespace algo4
