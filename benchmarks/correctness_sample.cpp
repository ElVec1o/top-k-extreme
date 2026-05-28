// Correctness probe for SampleFilterTopK.
//
// Top-K from the sample-and-filter is EXACT (not approximate) — the
// final nth_element runs on a real superset of the true top-K with
// high probability, and falls back to full nth_element otherwise.
// So we verify against std::sort ground truth.

#include "../src/sample_filter.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<uint64_t> truth_topk(std::vector<uint64_t> v, size_t K) {
    std::sort(v.begin(), v.end(), std::greater<uint64_t>());
    v.resize(K);
    std::sort(v.begin(), v.end());
    return v;
}

static bool check(const std::vector<uint64_t>& data, size_t K,
                  const char* label) {
    auto truth = truth_topk(data, K);
    algo4::SampleFilterTopK sf(K, 16384, 4.0);
    auto got = sf.run(data);
    std::sort(got.begin(), got.end());

    bool ok = (got == truth);
    std::printf("[%s K=%zu N=%zu]  %s\n", label, K, data.size(),
                ok ? "OK" : "FAIL");
    if (!ok) {
        size_t diffs = 0;
        for (size_t i = 0; i < std::min(got.size(), truth.size()); ++i)
            if (got[i] != truth[i]) ++diffs;
        std::printf("  diffs=%zu got_size=%zu truth_size=%zu\n",
                    diffs, got.size(), truth.size());
    }
    return ok;
}

int main() {
    std::mt19937_64 rng(42);
    // The interesting regime: K close to N
    {
        std::vector<uint64_t> v(5'000'000);
        for (auto& x : v) x = rng() | 1;
        check(v, 1000,    "uniform-5M");
        check(v, 10000,   "uniform-5M");
        check(v, 100000,  "uniform-5M");
        check(v, 1000000, "uniform-5M");   // the K=N/5 case
        check(v, 2000000, "uniform-5M");   // K=N/2.5
    }
    {
        std::vector<uint64_t> v(1'000'000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = uint64_t(i + 1);
        check(v, 1000,   "sorted-asc");
        check(v, 100000, "sorted-asc");
    }
    {
        std::vector<uint64_t> v(1'000'000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = uint64_t(v.size() - i);
        check(v, 1000,   "sorted-desc");
        check(v, 100000, "sorted-desc");
    }
    {
        std::vector<uint64_t> v(1'000'000);
        for (auto& x : v) x = (rng() % 1000) + 1;  // high duplicates
        check(v, 100,  "high-dup");
        check(v, 1000, "high-dup");
    }
    return 0;
}
