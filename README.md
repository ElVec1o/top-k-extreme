# Top-K Extreme — Streaming Top-K for uint64 on Apple M4

Three C++ primitives that together cover every K from 100 to 1M,
beating `std::priority_queue` by **1.3x at small K** and **23x at K=1M**.

Every number has an adversarial correctness probe behind it.

---

## Scoreboard

Median of 3 runs, 5M uniform random uint64, single M4 P-core (Mops/s, higher = better):

| K | std::priority_queue | std::nth_element (offline) | async-membrane | sample-filter | **radix-select** |
|---:|---:|---:|---:|---:|---:|
| 100 | 1,635 | 231 | **2,084** | 1,510 | 1,010 |
| 1,000 | 1,545 | 222 | **2,083** | 1,119 | 909 |
| 10,000 | 632 | 280 | 1,025 | **1,211** | 1,125 |
| 100,000 | 145 | 266 | 269 | 613 | **945** |
| 1,000,000 | 17 | 264 | 15 | 225 | **397** |

Per-row champion picks the right primitive for the regime:
async-membrane for tiny K, sample-filter for medium K, radix-select for large K.

---

## The three primitives

### 1. Async Membrane Top-K (`src/pipeline.hpp`)

Scalar threshold gates the hot path; items below the threshold die in
one register compare. Items above land in a double-buffer; a background
thread runs `std::nth_element` on each filled buffer to update the
threshold. Best at **K <= 1K**.

### 2. Sample-Filter Top-K (`src/sample_filter.hpp`)

Floyd-Rivest 1975. Take a 16K random sample, find a statistically
conservative threshold via `std::nth_element`, one sequential filter
pass, final `nth_element` on the survivors. Best at **K = 10K**.

### 3. Radix-Select Top-K (`src/radix_topk.hpp`)

4-way independent histogram on the top byte (256 buckets, L1-resident),
find the pivot bucket containing the K-th rank, emit items strictly
above pivot + scratch, `nth_element` on the scratch. O(N) with tight
sequential memory access. Best at **K >= 100K**.

---

## Correctness

Every primitive has an adversarial probe:
- `benchmarks/correctness_radix.cpp`
- `benchmarks/correctness_sample.cpp`
- `benchmarks/correctness_rank_bucket.cpp`

All pass on uniform / sorted-asc / sorted-desc / high-dup / K=N.

---

## Repo layout

```
src/
  pipeline.hpp           AsyncMembraneTopK + GaloisFilter
  sample_filter.hpp      Floyd-Rivest sample-then-filter Top-K
  radix_topk.hpp         Radix-select Top-K (the headline)
  rank_bucket.hpp        16-bit-bucket variant (tested, slower)
  fast_hash_set.hpp      Open-addressed integer hash set

benchmarks/
  bench.cpp              Multi-mode harness (throughput/sweep/latency/distribution)
  correctness_*.cpp      Adversarial probes

results/                 Captured benchmark runs
Makefile                 Builds bench (links abseil)
verify_and_bench.sh      One-shot: correctness probes + 3-run K sweep
```

## Quick start

```bash
./verify_and_bench.sh
./build/bench --mode throughput --n 10000000 --k 1000
./build/bench --mode sweep --n 5000000
```

## Dependencies

- C++17, Apple Clang or GCC
- [Abseil](https://abseil.io/) (for baseline hash set benchmarks)
- Apple M-series recommended (NEON codegen)
