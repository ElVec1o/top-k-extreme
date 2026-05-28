// Correctness probe — does AdaptiveAsyncTopK produce the exact top-K?
// Compare its output to a ground-truth std::sort, on uniform and adversarial
// inputs, across small/medium/large K.

#include "../src/pipeline.hpp"
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

template <class TK>
static bool check_impl(const std::vector<uint64_t>& data, size_t K,
                      const char* label, const char* algo) {
    auto truth = truth_topk(data, K);

    TK tk(K, 8192);
    for (uint64_t v : data) tk.push(v);
    auto got = tk.finalize();
    std::sort(got.begin(), got.end());

    bool ok = (got == truth);
    std::printf("[%s/%s K=%zu N=%zu]  %s",
                algo, label, K, data.size(), ok ? "OK" : "FAIL");
    if (!ok) {
        size_t diffs = 0;
        for (size_t i = 0; i < std::min(truth.size(), got.size()); ++i)
            if (truth[i] != got[i]) ++diffs;
        std::printf("  diffs=%zu got_size=%zu truth_size=%zu",
                    diffs, got.size(), truth.size());
    }
    std::printf("\n");
    return ok;
}

static bool check(const std::vector<uint64_t>& data, size_t K,
                  const char* label) {
    bool a = check_impl<algo4::AdaptiveAsyncTopK>(data, K, label, "Adaptive");
    bool b = check_impl<algo4::AsyncMembraneTopK>(data, K, label, "Async   ");
    bool ok = a && b;
    return ok;
}

int main() {
    std::mt19937_64 rng(42);

    // Small K, small N
    {
        std::vector<uint64_t> v(10'000);
        for (auto& x : v) x = rng() | 1;
        check(v, 10,   "uniform-small");
        check(v, 100,  "uniform-small");
        check(v, 1000, "uniform-small");
    }
    // Medium N
    {
        std::vector<uint64_t> v(1'000'000);
        for (auto& x : v) x = rng() | 1;
        check(v, 100,    "uniform-mid");
        check(v, 1000,   "uniform-mid");
        check(v, 10000,  "uniform-mid");
        check(v, 100000, "uniform-mid");
    }
    // The K=1M heap path (this is where adaptive switches)
    {
        std::vector<uint64_t> v(5'000'000);
        for (auto& x : v) x = rng() | 1;
        check(v, 1'000'000, "uniform-big-K");
    }
    // Adversarial: sorted ascending (every item breaches threshold)
    {
        std::vector<uint64_t> v(1'000'000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (uint64_t)(i + 1);
        check(v, 1000,   "sorted-asc");
        check(v, 100000, "sorted-asc");
    }
    // Adversarial: sorted descending (nothing breaches after first K)
    {
        std::vector<uint64_t> v(1'000'000);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = (uint64_t)(v.size() - i);
        check(v, 1000,   "sorted-desc");
        check(v, 100000, "sorted-desc");
    }
    // Many duplicates
    {
        std::vector<uint64_t> v(1'000'000);
        for (auto& x : v) x = (rng() % 1000) + 1;
        check(v, 100,  "high-dup");
        check(v, 1000, "high-dup");
    }
    return 0;
}
