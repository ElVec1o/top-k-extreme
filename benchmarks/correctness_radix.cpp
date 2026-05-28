// Correctness probe for radix_select_topk.

#include "../src/radix_topk.hpp"
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
    auto got = algo4::radix_select_topk(data, K);
    std::sort(got.begin(), got.end());

    bool ok = (got == truth);
    std::printf("[radix/%s K=%zu N=%zu]  %s\n", label, K, data.size(),
                ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("  got_size=%zu truth_size=%zu\n", got.size(), truth.size());
        size_t diffs = 0;
        for (size_t i = 0; i < std::min(got.size(), truth.size()); ++i)
            if (got[i] != truth[i]) ++diffs;
        std::printf("  diffs=%zu\n", diffs);
        // Show first divergence
        for (size_t i = 0; i < std::min(got.size(), truth.size()); ++i) {
            if (got[i] != truth[i]) {
                std::printf("  first diff @ i=%zu: got=%llu truth=%llu\n",
                            i, (unsigned long long)got[i],
                            (unsigned long long)truth[i]);
                break;
            }
        }
    }
    return ok;
}

int main() {
    std::mt19937_64 rng(42);
    {
        std::vector<uint64_t> v(5'000'000);
        for (auto& x : v) x = rng() | 1;
        check(v, 100,     "uniform");
        check(v, 1000,    "uniform");
        check(v, 10000,   "uniform");
        check(v, 100000,  "uniform");
        check(v, 1000000, "uniform");
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
        for (auto& x : v) x = (rng() % 1000) + 1;
        check(v, 100,  "high-dup");
        check(v, 1000, "high-dup");
    }
    // Edge: K equals N
    {
        std::vector<uint64_t> v(1000);
        for (auto& x : v) x = rng() | 1;
        check(v, 1000, "K-eq-N");
    }
    return 0;
}
