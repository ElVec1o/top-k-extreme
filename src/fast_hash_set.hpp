// fast_hash_set.hpp
//
// A hand-tuned open-addressed integer hash set. This exists as a "no
// excuses" baseline for the dedup competition — when our pipeline beats
// this one, the win isn't an artifact of std::unordered_set being slow.
//
// Layout:
//   - Power-of-two capacity, linear probing.
//   - Sentinel EMPTY = 0; callers must never insert 0. (Easy for the
//     benchmark workload since we hash random uint64.)
//   - 50% max load factor for fast probes.
//   - SplitMix64 mixer for hash.

#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace algo4 {

class FastHashSet {
public:
    static constexpr uint64_t EMPTY = 0;

    explicit FastHashSet(size_t expected_keys) {
        size_t cap = 16;
        while (cap < expected_keys * 2) cap <<= 1;
        slots_.assign(cap, EMPTY);
        mask_ = cap - 1;
    }

    // Returns true if newly inserted.
    inline __attribute__((always_inline)) bool insert(uint64_t k) {
        // Treat 0 specially — bench data never produces it.
        uint64_t h = mix(k);
        size_t i = h & mask_;
        for (;;) {
            uint64_t s = slots_[i];
            if (s == EMPTY) { slots_[i] = k; ++size_; maybe_grow(); return true; }
            if (s == k)     { return false; }
            i = (i + 1) & mask_;
        }
    }

    inline bool contains(uint64_t k) const {
        uint64_t h = mix(k);
        size_t i = h & mask_;
        for (;;) {
            uint64_t s = slots_[i];
            if (s == EMPTY) return false;
            if (s == k)     return true;
            i = (i + 1) & mask_;
        }
    }

    size_t size() const { return size_; }

private:
    static inline uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    void maybe_grow() {
        if (size_ * 2 <= slots_.size()) return;
        size_t new_cap = slots_.size() * 2;
        std::vector<uint64_t> old(new_cap, EMPTY);
        size_t new_mask = new_cap - 1;
        for (uint64_t s : slots_) {
            if (s == EMPTY) continue;
            size_t i = mix(s) & new_mask;
            while (old[i] != EMPTY) i = (i + 1) & new_mask;
            old[i] = s;
        }
        slots_.swap(old);
        mask_ = new_mask;
    }

    std::vector<uint64_t> slots_;
    size_t                mask_ = 0;
    size_t                size_ = 0;
};

} // namespace algo4
