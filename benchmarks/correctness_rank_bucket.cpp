// Correctness probe for StreamingRankBucket.
//
// We MUST verify this honestly. The prior "Adaptive heap" trick gave us
// a fake 25× win because it silently dropped 67% of items at K=1M. The
// rank-bucket has identical risk: a threshold-management bug would
// produce fast nonsense. Every benchmark number is meaningless until
// this passes on adversarial probes.

#include "../src/rank_bucket.hpp"
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
    algo4::StreamingRankBucket sb(K);
    auto got = sb.run(data);
    std::sort(got.begin(), got.end());

    bool ok = (got == truth);
    std::printf("[rank-bucket/%s K=%zu N=%zu]  %s\n",
                label, K, data.size(), ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("  got_size=%zu truth_size=%zu\n", got.size(), truth.size());
        size_t diffs = 0;
        for (size_t i = 0; i < std::min(got.size(), truth.size()); ++i)
            if (got[i] != truth[i]) ++diffs;
        std::printf("  diffs=%zu\n", diffs);
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
    bool all_ok = true;
    {
        std::vector<uint64_t> v(5'000'000);
        for (auto& x : v) x = rng() | 1;
        all_ok &= check(v, 100,     "uniform");
        all_ok &= check(v, 1000,    "uniform");
        all_ok &= check(v, 10000,   "uniform");
        all_ok &= check(v, 100000,  "uniform");
        all_ok &= check(v, 1000000, "uniform");
    }
    {
        std::vector<uint64_t> v(1'000'000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = uint64_t(i + 1);
        all_ok &= check(v, 1000,   "sorted-asc");
        all_ok &= check(v, 100000, "sorted-asc");
    }
    {
        std::vector<uint64_t> v(1'000'000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = uint64_t(v.size() - i);
        all_ok &= check(v, 1000,   "sorted-desc");
        all_ok &= check(v, 100000, "sorted-desc");
    }
    {
        std::vector<uint64_t> v(1'000'000);
        for (auto& x : v) x = (rng() % 1000) + 1;
        all_ok &= check(v, 100,  "high-dup");
        all_ok &= check(v, 1000, "high-dup");
    }
    {
        std::vector<uint64_t> v(1000);
        for (auto& x : v) x = rng() | 1;
        all_ok &= check(v, 1000, "K-eq-N");
    }
    std::printf("\n[rank-bucket overall] %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
