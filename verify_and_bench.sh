#!/usr/bin/env bash
# verify_and_bench.sh
#
# Runs after `/private/tmp` is cleared. Compiles every primitive,
# verifies correctness on adversarial probes, then runs the K-sweep
# 3 times for median reporting. Exits non-zero on the first failure.
#
# Usage: ./verify_and_bench.sh
#
# Honesty gate: any new "fast" number that arrives without a passing
# correctness probe is rejected by this script. We don't get fooled
# again by the trailing-store bug.

set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)"
mkdir -p build results

CXXFLAGS_BASE=(-O3 -std=c++17 -march=native -Wall -Wextra -pthread)
CXXFLAGS_BENCH=("${CXXFLAGS_BASE[@]}"
                -I/opt/homebrew/include)
LDFLAGS_BENCH=(-L/opt/homebrew/lib
               -labsl_hash -labsl_city -labsl_raw_hash_set
               -labsl_hashtablez_sampler -labsl_raw_logging_internal
               -labsl_synchronization -labsl_base -labsl_throw_delegate
               -labsl_strings)

echo "=== [1/5] Build bench harness ==="
c++ "${CXXFLAGS_BENCH[@]}" benchmarks/bench.cpp \
    -o build/bench "${LDFLAGS_BENCH[@]}"

echo "=== [2/5] Correctness: radix-select ==="
c++ "${CXXFLAGS_BASE[@]}" benchmarks/correctness_radix.cpp \
    -o build/correctness_radix
./build/correctness_radix

echo "=== [3/5] Correctness: sample-filter ==="
c++ "${CXXFLAGS_BASE[@]}" benchmarks/correctness_sample.cpp \
    -o build/correctness_sample
./build/correctness_sample

echo "=== [4/5] Correctness: rank-bucket ==="
c++ "${CXXFLAGS_BASE[@]}" benchmarks/correctness_rank_bucket.cpp \
    -o build/correctness_rank_bucket
./build/correctness_rank_bucket

echo "=== [5/5] K-sweep × 3 ==="
for run in 1 2 3; do
    echo "--- run $run ---"
    ./build/bench --mode sweep --n 5000000 --dup 0.20 2>&1 \
      | tee "results/sweep_run${run}.txt" \
      | grep -E '(--- K=|^    our-radix|^    our-rank-bucket|^    our-sample-filter|^    stdlib-heap|^    stdlib-nth_element)'
done

echo ""
echo "=== Done. Results in results/sweep_run{1,2,3}.txt ==="
echo ""
echo "Quick check — is rank-bucket faster than radix at K=100K and K=1M?"
echo "  (compare the two primitives side-by-side)"
for k in 100000 1000000; do
    echo "K=$k:"
    for r in 1 2 3; do
        grep -A20 "K=$k ---" "results/sweep_run${r}.txt" 2>/dev/null \
          | grep -E 'our-radix|our-rank-bucket' \
          | head -2
    done
done
