// bench.cpp — brutal head-to-head benchmark suite.
//
// Modes:
//   throughput          single (N,K,dist) run, all algos, Mops/s
//   sweep               sweep K over {100, 1000, 10000, 100000, 1000000}
//   latency             per-batch latency histogram for tail analysis
//   distribution        run all algos on uniform / sorted / zipf / heavy-tail
//
// Competitors:
//   Top-K-only:
//     [stdlib-heap]            std::priority_queue bounded to K
//     [stdlib-sort]            offline std::sort + take last K
//     [our-membrane]           EVT membrane + nth_element flush (sync)
//     [our-async]              async double-buffered membrane (V3)
//   Dedup + Top-K:
//     [dedup-stdlib-heap]      std::unordered_set + priority_queue
//     [dedup-absl-heap]        absl::flat_hash_set + priority_queue
//     [dedup-fastset-heap]     hand-tuned linear-probing set + priority_queue
//     [our-pipeline-RANK]      rank-then-filter (membrane → galois → push)
//     [our-pipeline-FILTER]    filter-then-rank (worst-case ordering)
//
// Output: JSON on stdout (parsed by the GUI), human summary on stderr.

#include "../src/pipeline.hpp"
#include "../src/fast_hash_set.hpp"
#include "../src/sample_filter.hpp"
#include "../src/radix_topk.hpp"
#include "../src/rank_bucket.hpp"

#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using clk = std::chrono::high_resolution_clock;
using dms = std::chrono::duration<double, std::milli>;
using dns = std::chrono::duration<double, std::nano>;

// ---------------------------------------------------------------------------
// Workload generators
// ---------------------------------------------------------------------------
enum class Dist { Uniform, SortedAsc, SortedDesc, Zipf, Lognormal };

static const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Uniform:    return "uniform";
        case Dist::SortedAsc:  return "sorted-asc";
        case Dist::SortedDesc: return "sorted-desc";
        case Dist::Zipf:       return "zipf";
        case Dist::Lognormal:  return "lognormal";
    }
    return "?";
}

static Dist parse_dist(const std::string& s) {
    if (s == "sorted-asc")  return Dist::SortedAsc;
    if (s == "sorted-desc") return Dist::SortedDesc;
    if (s == "zipf")        return Dist::Zipf;
    if (s == "lognormal")   return Dist::Lognormal;
    return Dist::Uniform;
}

static std::vector<uint64_t> make_stream(size_t n, double dup_ratio,
                                         Dist dist, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);

    auto sample = [&]() -> uint64_t {
        switch (dist) {
            case Dist::Uniform:
                return rng() | 1;  // never 0 (sentinel for FastHashSet)
            case Dist::Lognormal: {
                std::lognormal_distribution<double> d(0.0, 2.0);
                double x = d(rng);
                return (uint64_t)(x * 1e9) | 1;
            }
            case Dist::Zipf: {
                // Crude Zipf(s≈1.2) over 1e7 ids — enough mass on heavy
                // hitters to stress dedup paths.
                std::uniform_real_distribution<double> u(0.0, 1.0);
                double r = u(rng);
                uint64_t rank = (uint64_t)std::pow(1.0 / (r + 1e-12), 1.0 / 1.2);
                return (rank % 10'000'000ULL) | 1;
            }
            default: return rng() | 1;
        }
    };

    size_t uniq = std::max<size_t>(1, (size_t)((1.0 - dup_ratio) * n));
    for (size_t i = 0; i < uniq; ++i) v[i] = sample();
    for (size_t i = uniq; i < n; ++i) v[i] = v[rng() % uniq];

    if (dist == Dist::SortedAsc) {
        std::sort(v.begin(), v.end());
    } else if (dist == Dist::SortedDesc) {
        std::sort(v.begin(), v.end(), std::greater<uint64_t>());
    } else {
        std::shuffle(v.begin(), v.end(), rng);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Algorithm runners (each takes data + K, returns elapsed ms)
// ---------------------------------------------------------------------------
template <class F>
static double timed(F&& f) {
    auto t0 = clk::now();
    f();
    auto t1 = clk::now();
    return dms(t1 - t0).count();
}

static double run_stdlib_heap(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> pq;
        for (uint64_t v : data) {
            if (pq.size() < K) pq.push(v);
            else if (v > pq.top()) { pq.pop(); pq.push(v); }
        }
    });
}

static double run_stdlib_sort(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        std::vector<uint64_t> v = data;
        std::nth_element(v.begin(), v.begin() + K - 1, v.end(),
                         std::greater<uint64_t>());
    });
}

static double run_membrane(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        algo4::AsyncMembraneTopK tk(K, 4096);
        for (uint64_t v : data) tk.push(v);
        (void)tk.finalize();
    });
}

static double run_async(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        algo4::AsyncMembraneTopK tk(K, 8192);
        for (uint64_t v : data) tk.push(v);
        (void)tk.finalize();
    });
}

static double run_adaptive(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        algo4::AdaptiveAsyncTopK tk(K, 8192);
        for (uint64_t v : data) tk.push(v);
        (void)tk.finalize();
    });
}

static double run_sample_filter(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        algo4::SampleFilterTopK sf(K, /*M=*/16384, /*oversample=*/4.0);
        auto r = sf.run(data);
        (void)r;
    });
}

static double run_radix(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        auto r = algo4::radix_select_topk(data, K);
        (void)r;
    });
}

static double run_rank_bucket(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        algo4::StreamingRankBucket sb(K);
        auto r = sb.run(data);
        (void)r;
    });
}

static double run_dedup_stdlib(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        std::unordered_set<uint64_t> seen;
        seen.reserve(data.size() * 2);
        std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> pq;
        for (uint64_t v : data) {
            if (!seen.insert(v).second) continue;
            if (pq.size() < K) pq.push(v);
            else if (v > pq.top()) { pq.pop(); pq.push(v); }
        }
    });
}

static double run_dedup_absl(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        absl::flat_hash_set<uint64_t> seen;
        seen.reserve(data.size() * 2);
        std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> pq;
        for (uint64_t v : data) {
            if (!seen.insert(v).second) continue;
            if (pq.size() < K) pq.push(v);
            else if (v > pq.top()) { pq.pop(); pq.push(v); }
        }
    });
}

static double run_dedup_fastset(const std::vector<uint64_t>& data, size_t K) {
    return timed([&] {
        algo4::FastHashSet seen(data.size() * 2);
        std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> pq;
        for (uint64_t v : data) {
            if (!seen.insert(v)) continue;
            if (pq.size() < K) pq.push(v);
            else if (v > pq.top()) { pq.pop(); pq.push(v); }
        }
    });
}

static double run_pipeline_rank(const std::vector<uint64_t>& data, size_t K,
                                const algo4::GaloisFilter& gf) {
    return timed([&] {
        algo4::AsyncMembraneTopK tk(K, 8192);
        algo4::RankFirstPipeline p{const_cast<algo4::GaloisFilter&>(gf), tk};
        for (uint64_t v : data) p.feed(v);
        (void)tk.finalize();
    });
}

static double run_pipeline_filter(const std::vector<uint64_t>& data, size_t K,
                                  const algo4::GaloisFilter& gf) {
    return timed([&] {
        algo4::AsyncMembraneTopK tk(K, 8192);
        algo4::FilterFirstPipeline p{const_cast<algo4::GaloisFilter&>(gf), tk};
        for (uint64_t v : data) p.feed(v);
        (void)tk.finalize();
    });
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
struct Algo {
    std::string name;
    bool        does_dedup;
    // Closure that runs the algo and returns ms. (gf only used by pipelines.)
    using Fn = std::function<double(const std::vector<uint64_t>&, size_t,
                                    const algo4::GaloisFilter&)>;
    Fn run;
};

static std::vector<Algo> all_algos() {
    return {
        {"stdlib-heap",         false, [](auto& d, auto K, auto&) { return run_stdlib_heap(d, K); }},
        {"stdlib-nth_element",  false, [](auto& d, auto K, auto&) { return run_stdlib_sort(d, K); }},
        {"our-membrane",        false, [](auto& d, auto K, auto&) { return run_membrane(d, K); }},
        {"our-async",           false, [](auto& d, auto K, auto&) { return run_async(d, K); }},
        {"our-adaptive",        false, [](auto& d, auto K, auto&) { return run_adaptive(d, K); }},
        {"our-sample-filter",   false, [](auto& d, auto K, auto&) { return run_sample_filter(d, K); }},
        {"our-radix",           false, [](auto& d, auto K, auto&) { return run_radix(d, K); }},
        {"our-rank-bucket",     false, [](auto& d, auto K, auto&) { return run_rank_bucket(d, K); }},
        {"dedup-stdlib-heap",   true,  [](auto& d, auto K, auto&) { return run_dedup_stdlib(d, K); }},
        {"dedup-absl-heap",     true,  [](auto& d, auto K, auto&) { return run_dedup_absl(d, K); }},
        {"dedup-fastset-heap",  true,  [](auto& d, auto K, auto&) { return run_dedup_fastset(d, K); }},
        {"our-pipeline-RANK",   true,  [](auto& d, auto K, auto& g) { return run_pipeline_rank(d, K, g); }},
        {"our-pipeline-FILTER", true,  [](auto& d, auto K, auto& g) { return run_pipeline_filter(d, K, g); }},
    };
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
static void emit_kv(const std::string& k, const std::string& v, bool last=false) {
    std::printf("  \"%s\": \"%s\"%s\n", k.c_str(), v.c_str(), last ? "" : ",");
}
static void emit_kv_num(const std::string& k, double v, bool last=false) {
    std::printf("  \"%s\": %.6f%s\n", k.c_str(), v, last ? "" : ",");
}

static algo4::GaloisFilter build_filter(const std::vector<uint64_t>& data) {
    algo4::GaloisFilter gf(data.size());
    size_t sample = std::min<size_t>(data.size() / 4, 5'000'000);
    std::vector<uint64_t> keys(data.begin(), data.begin() + sample);
    gf.build(keys);
    return gf;
}

static void mode_throughput(size_t N, size_t K, double dup, Dist dist) {
    auto data = make_stream(N, dup, dist);
    auto gf   = build_filter(data);
    auto algos = all_algos();

    std::fprintf(stderr, "[throughput] N=%zu K=%zu dup=%.2f dist=%s\n",
                 N, K, dup, dist_name(dist));

    std::printf("{\n");
    emit_kv("mode", "throughput");
    emit_kv("dist", dist_name(dist));
    emit_kv_num("n", (double)N);
    emit_kv_num("k", (double)K);
    emit_kv_num("dup_ratio", dup);
    std::printf("  \"results\": [\n");
    for (size_t i = 0; i < algos.size(); ++i) {
        double ms = algos[i].run(data, K, gf);
        double mops = (double)N / (ms / 1000.0) / 1e6;
        std::fprintf(stderr, "  %-24s  %9.2f ms   %10.1f Mops/s%s\n",
                     algos[i].name.c_str(), ms, mops,
                     algos[i].does_dedup ? "   [dedup]" : "");
        std::printf("    {\"name\": \"%s\", \"dedup\": %s, "
                    "\"ms\": %.4f, \"mops\": %.4f}%s\n",
                    algos[i].name.c_str(),
                    algos[i].does_dedup ? "true" : "false",
                    ms, mops, i + 1 < algos.size() ? "," : "");
    }
    std::printf("  ]\n}\n");
}

static void mode_sweep(size_t N, double dup, Dist dist) {
    auto data = make_stream(N, dup, dist);
    auto gf   = build_filter(data);
    auto algos = all_algos();
    std::vector<size_t> Ks = {100, 1000, 10000, 100000, 1000000};

    std::fprintf(stderr, "[sweep] N=%zu dup=%.2f dist=%s K=", N, dup, dist_name(dist));
    for (auto k : Ks) std::fprintf(stderr, "%zu ", k);
    std::fprintf(stderr, "\n");

    std::printf("{\n");
    emit_kv("mode", "sweep");
    emit_kv("dist", dist_name(dist));
    emit_kv_num("n", (double)N);
    emit_kv_num("dup_ratio", dup);
    std::printf("  \"runs\": [\n");
    bool first_outer = true;
    for (size_t K : Ks) {
        if (K * 2 > N) continue;
        std::fprintf(stderr, "  --- K=%zu ---\n", K);
        if (!first_outer) std::printf(",\n");
        first_outer = false;
        std::printf("    {\"k\": %zu, \"results\": [", K);
        for (size_t i = 0; i < algos.size(); ++i) {
            double ms = algos[i].run(data, K, gf);
            double mops = (double)N / (ms / 1000.0) / 1e6;
            std::fprintf(stderr, "    %-24s  %9.2f ms   %10.1f Mops/s\n",
                         algos[i].name.c_str(), ms, mops);
            std::printf("%s{\"name\":\"%s\",\"dedup\":%s,\"ms\":%.4f,\"mops\":%.4f}",
                        i ? "," : "",
                        algos[i].name.c_str(),
                        algos[i].does_dedup ? "true" : "false",
                        ms, mops);
        }
        std::printf("]}");
    }
    std::printf("\n  ]\n}\n");
}

// ---------------------------------------------------------------------------
// Latency mode — block-timed histogram so per-item overhead doesn't dominate.
// Each algorithm sees the stream in 1024-item blocks; we record ns/item for
// each block, then compute p50/p90/p99/p99.9/max.
// ---------------------------------------------------------------------------
struct LatStats {
    double p50, p90, p99, p999, max, mean;
};

static LatStats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double p) {
        size_t i = std::min<size_t>(v.size() - 1, (size_t)(p * v.size()));
        return v[i];
    };
    double sum = 0;
    for (double x : v) sum += x;
    return {q(0.50), q(0.90), q(0.99), q(0.999), v.back(), sum / v.size()};
}

template <class Pusher>
static LatStats latency_block_timed(const std::vector<uint64_t>& data,
                                    Pusher&& push_one,
                                    size_t block = 1024) {
    std::vector<double> samples;
    samples.reserve(data.size() / block);
    for (size_t i = 0; i + block <= data.size(); i += block) {
        auto t0 = clk::now();
        for (size_t j = 0; j < block; ++j) push_one(data[i + j]);
        auto t1 = clk::now();
        samples.push_back(dns(t1 - t0).count() / block);
    }
    return compute_stats(samples);
}

static void mode_latency(size_t N, size_t K, double dup, Dist dist) {
    auto data = make_stream(N, dup, dist);
    auto gf   = build_filter(data);

    std::fprintf(stderr, "[latency] N=%zu K=%zu dup=%.2f dist=%s\n",
                 N, K, dup, dist_name(dist));

    struct LatAlgo {
        std::string name;
        LatStats    s;
    };
    std::vector<LatAlgo> out;

    // stdlib heap
    {
        std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> pq;
        auto s = latency_block_timed(data, [&](uint64_t v) {
            if (pq.size() < K) pq.push(v);
            else if (v > pq.top()) { pq.pop(); pq.push(v); }
        });
        out.push_back({"stdlib-heap", s});
    }
    // our async
    {
        algo4::AsyncMembraneTopK tk(K, 8192);
        auto s = latency_block_timed(data, [&](uint64_t v) { tk.push(v); });
        out.push_back({"our-async", s});
    }
    // dedup absl + heap
    {
        absl::flat_hash_set<uint64_t> seen;
        seen.reserve(data.size() * 2);
        std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<>> pq;
        auto s = latency_block_timed(data, [&](uint64_t v) {
            if (!seen.insert(v).second) return;
            if (pq.size() < K) pq.push(v);
            else if (v > pq.top()) { pq.pop(); pq.push(v); }
        });
        out.push_back({"dedup-absl-heap", s});
    }
    // our rank-first
    {
        algo4::AsyncMembraneTopK tk(K, 8192);
        algo4::RankFirstPipeline p{gf, tk};
        auto s = latency_block_timed(data, [&](uint64_t v) { p.feed(v); });
        out.push_back({"our-pipeline-RANK", s});
    }

    std::printf("{\n");
    emit_kv("mode", "latency");
    emit_kv("dist", dist_name(dist));
    emit_kv_num("n", (double)N);
    emit_kv_num("k", (double)K);
    emit_kv_num("dup_ratio", dup);
    std::printf("  \"results\": [\n");
    for (size_t i = 0; i < out.size(); ++i) {
        const auto& a = out[i];
        std::fprintf(stderr,
            "  %-24s  p50=%.1f p90=%.1f p99=%.1f p99.9=%.1f max=%.1f ns/item\n",
            a.name.c_str(), a.s.p50, a.s.p90, a.s.p99, a.s.p999, a.s.max);
        std::printf("    {\"name\":\"%s\",\"p50\":%.4f,\"p90\":%.4f,"
                    "\"p99\":%.4f,\"p999\":%.4f,\"max\":%.4f,\"mean\":%.4f}%s\n",
                    a.name.c_str(), a.s.p50, a.s.p90, a.s.p99, a.s.p999,
                    a.s.max, a.s.mean, i + 1 < out.size() ? "," : "");
    }
    std::printf("  ]\n}\n");
}

static void mode_distribution(size_t N, size_t K, double dup) {
    auto algos = all_algos();
    std::vector<Dist> dists = {Dist::Uniform, Dist::SortedAsc, Dist::SortedDesc,
                                Dist::Zipf, Dist::Lognormal};

    std::printf("{\n");
    emit_kv("mode", "distribution");
    emit_kv_num("n", (double)N);
    emit_kv_num("k", (double)K);
    emit_kv_num("dup_ratio", dup);
    std::printf("  \"runs\": [\n");
    bool first = true;
    for (Dist d : dists) {
        auto data = make_stream(N, dup, d);
        auto gf   = build_filter(data);
        std::fprintf(stderr, "[dist=%s]\n", dist_name(d));
        if (!first) std::printf(",\n");
        first = false;
        std::printf("    {\"dist\":\"%s\",\"results\":[", dist_name(d));
        for (size_t i = 0; i < algos.size(); ++i) {
            double ms = algos[i].run(data, K, gf);
            double mops = (double)N / (ms / 1000.0) / 1e6;
            std::fprintf(stderr, "  %-24s  %9.2f ms   %10.1f Mops/s\n",
                         algos[i].name.c_str(), ms, mops);
            std::printf("%s{\"name\":\"%s\",\"dedup\":%s,\"ms\":%.4f,\"mops\":%.4f}",
                        i ? "," : "",
                        algos[i].name.c_str(),
                        algos[i].does_dedup ? "true" : "false",
                        ms, mops);
        }
        std::printf("]}");
    }
    std::printf("\n  ]\n}\n");
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string mode = "throughput";
    size_t N = 10'000'000, K = 1000;
    double dup = 0.20;
    Dist dist = Dist::Uniform;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* err) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s\n", err); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--mode") mode = next("missing --mode value");
        else if (a == "--n")    N    = std::stoull(next("missing --n"));
        else if (a == "--k")    K    = std::stoull(next("missing --k"));
        else if (a == "--dup")  dup  = std::stod(next("missing --dup"));
        else if (a == "--dist") dist = parse_dist(next("missing --dist"));
        else if (a == "--help") {
            std::fprintf(stderr,
                "usage: bench --mode {throughput|sweep|latency|distribution}\n"
                "             [--n N] [--k K] [--dup D] [--dist {uniform|sorted-asc|sorted-desc|zipf|lognormal}]\n");
            return 0;
        }
    }

    if      (mode == "throughput")   mode_throughput(N, K, dup, dist);
    else if (mode == "sweep")        mode_sweep(N, dup, dist);
    else if (mode == "latency")      mode_latency(N, K, dup, dist);
    else if (mode == "distribution") mode_distribution(N, K, dup);
    else { std::fprintf(stderr, "unknown mode: %s\n", mode.c_str()); return 2; }
    return 0;
}
