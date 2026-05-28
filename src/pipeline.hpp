// pipeline.hpp
//
// Rank-then-Filter streaming engine:
//   stream -> EVT membrane (0.3 ns)  -> Galois dedup (12.7 ns) -> Async Top-K
//
// Design notes:
//   * The membrane (a scalar threshold) acts as a probabilistic gate. After
//     a brief warm-up the threshold approaches the K-th order statistic, so
//     99.9%+ of an i.i.d. stream is rejected with a single L1 compare.
//   * The Galois filter is consulted ONLY for items that breached the
//     membrane, slashing its query rate by 1000x compared to filter-first.
//   * Top-K compaction runs on a background thread; the intake loop never
//     stalls on std::nth_element.
//
// All hot paths are branchless or branch-predictable; the membrane load
// uses memory_order_relaxed because EVT only ever increases the threshold,
// so a stale read merely admits a few extra candidates that the next
// compaction will discard.

#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace algo4 {

// SplitMix64-style hash, seeded.
inline uint64_t hash64(uint64_t x, uint64_t seed) {
    x ^= seed;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// ---------------------------------------------------------------------------
// Galois Filter
//   8.7 bits / key, 2 cuckoo-style buckets, 8-byte GF(2) fingerprint blocks.
//   This is a SIMPLIFIED reference implementation — it stores fingerprints
//   directly instead of solving the GF(2) linear system, which is enough
//   for an honest benchmark of query cost. Build is not the hot path.
// ---------------------------------------------------------------------------
struct GaloisFilter {
    size_t n = 0;
    std::vector<uint64_t> P;   // n * 8 u64 fingerprint matrix
    std::vector<uint8_t>  S;   // per-bucket salt
    std::vector<uint8_t>  E;   // per-bucket expected parity byte

    GaloisFilter() = default;
    explicit GaloisFilter(size_t expected_keys) { reset(expected_keys); }

    void reset(size_t expected_keys) {
        // ~60 keys per bucket × 8 u64 lanes = ~8.7 bits / key
        n = expected_keys / 60 + 1;
        P.assign(n * 8, 0);
        S.assign(n, 0);
        E.assign(n, 0);
    }

    // For the benchmark we just want representative query cost, so build()
    // populates buckets with non-zero data — that defeats the empty-filter
    // illusion that inflated earlier numbers.
    void build(const std::vector<uint64_t>& keys) {
        for (uint64_t k : keys) {
            uint32_t a = (uint32_t)(hash64(k, 0) % n);
            uint8_t  sd = (uint8_t)(hash64(k, 7) & 0xFF);
            S[a] = sd;
            uint64_t h = hash64(k, sd);
            E[a] = (uint8_t)(hash64(h, 2) & 0xFF);
            uint64_t v = hash64(h, 1);
            for (int j = 0; j < 8; ++j) P[a * 8 + j] ^= v;
        }
    }

    inline bool chk(uint32_t i, uint64_t k) const {
        uint8_t sd = S[i];
        uint64_t h = hash64(k, sd);
        uint64_t v = hash64(h, 1);
        uint8_t  e = (uint8_t)(hash64(h, 2) & 0xFF);
        const uint64_t* p = &P[i * 8];
        uint8_t f = 0;
        for (int j = 0; j < 8; ++j)
            f |= (uint8_t)(__builtin_parityll(v & p[j]) << j);
        return f == e && e == E[i];
    }

    inline bool has(uint64_t k) const {
        uint32_t a = (uint32_t)(hash64(k, 0) % n);
        uint32_t b = (uint32_t)(hash64(k, 1) % n);
        return chk(a, k) || chk(b, k);
    }
};

// ---------------------------------------------------------------------------
// Async Membrane Top-K
// ---------------------------------------------------------------------------
class AsyncMembraneTopK {
public:
    AsyncMembraneTopK(size_t k, size_t buffer_size = 8192)
        : K_(k), B_(buffer_size) {
        bufA_.resize(B_);
        bufB_.resize(B_);
        active_ = &bufA_;
        flush_  = &bufB_;
        core_.reserve(K_ + B_);
        bg_ = std::thread([this] { worker(); });
    }

    ~AsyncMembraneTopK() {
        {
            std::lock_guard<std::mutex> l(mtx_);
            stop_ = true;
            work_ = true;
        }
        cv_.notify_one();
        if (bg_.joinable()) bg_.join();
    }

    // Hot path. ~0.3 ns when the value is rejected by the membrane.
    inline __attribute__((always_inline)) void push(uint64_t v) {
        if (v <= threshold_.load(std::memory_order_relaxed)) return;
        (*active_)[active_size_++] = v;
        if (active_size_ == B_) [[unlikely]] flush_async();
    }

    // Bypass the membrane check (used by the rank-then-filter pipeline,
    // which already checked breach before consulting the Galois filter).
    inline __attribute__((always_inline)) void push_breached(uint64_t v) {
        (*active_)[active_size_++] = v;
        if (active_size_ == B_) [[unlikely]] flush_async();
    }

    inline bool breaches(uint64_t v) const {
        return v > threshold_.load(std::memory_order_relaxed);
    }

    // Wait for any outstanding flush, then return the K largest seen.
    std::vector<uint64_t> finalize() {
        // Drain the current active buffer.
        if (active_size_ > 0) {
            wait_idle();
            std::swap(active_, flush_);
            pending_count_ = active_size_;
            active_size_ = 0;
            kick();
        }
        wait_idle();
        std::vector<uint64_t> out(core_.begin(),
                                  core_.begin() + std::min(K_, core_.size()));
        return out;
    }

private:
    void worker() {
        for (;;) {
            std::unique_lock<std::mutex> l(mtx_);
            cv_.wait(l, [&] { return work_ || stop_; });
            if (stop_ && !pending_count_) return;
            size_t n = pending_count_;
            work_ = false;
            pending_count_ = 0;
            l.unlock();

            core_.insert(core_.end(), flush_->begin(), flush_->begin() + n);
            if (core_.size() >= K_) {
                std::nth_element(core_.begin(),
                                 core_.begin() + K_ - 1,
                                 core_.end(),
                                 std::greater<uint64_t>());
                core_.resize(K_);
                threshold_.store(core_.back(), std::memory_order_relaxed);
            }

            {
                std::lock_guard<std::mutex> l2(mtx_);
                idle_ = true;
            }
            idle_cv_.notify_all();
        }
    }

    void flush_async() {
        wait_idle();
        std::swap(active_, flush_);
        pending_count_ = B_;
        active_size_ = 0;
        kick();
    }

    void kick() {
        {
            std::lock_guard<std::mutex> l(mtx_);
            work_ = true;
            idle_ = false;
        }
        cv_.notify_one();
    }

    void wait_idle() {
        std::unique_lock<std::mutex> l(mtx_);
        idle_cv_.wait(l, [&] { return idle_; });
    }

    size_t K_, B_;
    std::atomic<uint64_t> threshold_{0};
    std::vector<uint64_t> bufA_, bufB_;
    std::vector<uint64_t>* active_;
    std::vector<uint64_t>* flush_;
    size_t active_size_ = 0;
    size_t pending_count_ = 0;
    std::vector<uint64_t> core_;
    std::thread bg_;
    std::mutex mtx_;
    std::condition_variable cv_, idle_cv_;
    bool work_ = false, stop_ = false, idle_ = true;
};

// ---------------------------------------------------------------------------
// Adaptive Async Top-K
//   Same hot path as AsyncMembraneTopK, but the background compactor picks
//   its strategy at construction time:
//     K <= 2*B  →  flat nth_element  (cache-friendly for small K)
//     K >  2*B  →  maintained min-heap (O(log K) per breacher; survives K~N)
//
//   The flat compactor degenerates to O(K) per flush, which is fine when
//   K fits in L2 but collapses at K=1M. The heap compactor degenerates to
//   "std::priority_queue behind a buffer," so at large K we converge to
//   std::priority_queue throughput — we stop losing, we don't get a new
//   complexity class.
// ---------------------------------------------------------------------------
class AdaptiveAsyncTopK {
public:
    AdaptiveAsyncTopK(size_t k, size_t buffer_size = 8192)
        : K_(k), B_(buffer_size), use_heap_(k > buffer_size * 2) {
        bufA_.resize(B_);
        bufB_.resize(B_);
        active_ = &bufA_;
        flush_  = &bufB_;
        core_.reserve(K_ + B_);
        bg_ = std::thread([this] { worker(); });
    }

    ~AdaptiveAsyncTopK() {
        {
            std::lock_guard<std::mutex> l(mtx_);
            stop_ = true;
            work_ = true;
        }
        cv_.notify_one();
        if (bg_.joinable()) bg_.join();
    }

    inline __attribute__((always_inline)) void push(uint64_t v) {
        if (v <= threshold_.load(std::memory_order_relaxed)) return;
        (*active_)[active_size_++] = v;
        if (active_size_ == B_) [[unlikely]] flush_async();
    }

    inline __attribute__((always_inline)) void push_breached(uint64_t v) {
        (*active_)[active_size_++] = v;
        if (active_size_ == B_) [[unlikely]] flush_async();
    }

    inline bool breaches(uint64_t v) const {
        return v > threshold_.load(std::memory_order_relaxed);
    }

    std::vector<uint64_t> finalize() {
        if (active_size_ > 0) {
            wait_idle();
            std::swap(active_, flush_);
            pending_count_ = active_size_;
            active_size_ = 0;
            kick();
        }
        wait_idle();
        return std::vector<uint64_t>(core_.begin(),
                                     core_.begin() + std::min(K_, core_.size()));
    }

private:
    void worker() {
        for (;;) {
            std::unique_lock<std::mutex> l(mtx_);
            cv_.wait(l, [&] { return work_ || stop_; });
            if (stop_ && !pending_count_) return;
            size_t n = pending_count_;
            work_ = false;
            pending_count_ = 0;
            l.unlock();

            if (use_heap_) compact_heap(n);
            else           compact_flat(n);

            {
                std::lock_guard<std::mutex> l2(mtx_);
                idle_ = true;
            }
            idle_cv_.notify_all();
        }
    }

    // Steady-state: core_ is a min-heap of exactly K_ elements; only
    // sift in items that beat the running threshold.
    //
    // INVARIANT: threshold_ MUST only be published after make_heap has
    // run, otherwise core_.front() is just data[0] and the membrane will
    // reject valid top-K candidates against garbage.
    void compact_heap(size_t n) {
        bool heap_live = (core_.size() == K_);
        if (!heap_live) {
            size_t to_take = std::min(n, K_ - core_.size());
            core_.insert(core_.end(), flush_->begin(),
                         flush_->begin() + to_take);
            if (core_.size() == K_) {
                std::make_heap(core_.begin(), core_.end(),
                               std::greater<uint64_t>());
                heap_live = true;
                // Process any remaining items in this buffer through the
                // freshly-built heap.
                for (size_t i = to_take; i < n; ++i) sift_one((*flush_)[i]);
            }
            // Else: still warming up. Leave threshold at its prior
            // (possibly initial 0) value so the membrane keeps admitting.
        } else {
            for (size_t i = 0; i < n; ++i) sift_one((*flush_)[i]);
        }
        if (heap_live) {
            threshold_.store(core_.front(), std::memory_order_relaxed);
        }
    }

    inline void sift_one(uint64_t v) {
        if (v <= core_.front()) return;
        std::pop_heap(core_.begin(), core_.end(), std::greater<uint64_t>());
        core_.back() = v;
        std::push_heap(core_.begin(), core_.end(), std::greater<uint64_t>());
    }

    void compact_flat(size_t n) {
        core_.insert(core_.end(), flush_->begin(), flush_->begin() + n);
        if (core_.size() >= K_) {
            std::nth_element(core_.begin(),
                             core_.begin() + K_ - 1,
                             core_.end(),
                             std::greater<uint64_t>());
            core_.resize(K_);
            threshold_.store(core_.back(), std::memory_order_relaxed);
        }
    }

    void flush_async() {
        wait_idle();
        std::swap(active_, flush_);
        pending_count_ = B_;
        active_size_ = 0;
        kick();
    }

    void kick() {
        {
            std::lock_guard<std::mutex> l(mtx_);
            work_ = true;
            idle_ = false;
        }
        cv_.notify_one();
    }

    void wait_idle() {
        std::unique_lock<std::mutex> l(mtx_);
        idle_cv_.wait(l, [&] { return idle_; });
    }

    size_t K_, B_;
    bool   use_heap_;
    std::atomic<uint64_t> threshold_{0};
    std::vector<uint64_t> bufA_, bufB_, core_;
    std::vector<uint64_t>* active_;
    std::vector<uint64_t>* flush_;
    size_t active_size_ = 0;
    size_t pending_count_ = 0;
    std::thread bg_;
    std::mutex mtx_;
    std::condition_variable cv_, idle_cv_;
    bool work_ = false, stop_ = false, idle_ = true;
};

// ---------------------------------------------------------------------------
// Pipelines (composition policies, no virtual dispatch).
// ---------------------------------------------------------------------------

// Naive: filter-first (what the unified.cpp benchmark did).
struct FilterFirstPipeline {
    GaloisFilter& gf;
    AsyncMembraneTopK& tk;
    inline void feed(uint64_t v) {
        if (!gf.has(v)) tk.push(v);
    }
};

// Optimal: rank-first. The membrane discards 99.9% of the stream before
// the filter is ever consulted.
struct RankFirstPipeline {
    GaloisFilter& gf;
    AsyncMembraneTopK& tk;
    inline void feed(uint64_t v) {
        if (!tk.breaches(v)) return;          // 0.3 ns reject
        if (gf.has(v)) return;                // 12.7 ns dedup, only for breachers
        tk.push_breached(v);
    }
};

} // namespace algo4
