CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra -pthread \
            -I/opt/homebrew/include
LDFLAGS  ?= -L/opt/homebrew/lib -labsl_hash -labsl_city \
            -labsl_raw_hash_set -labsl_hashtablez_sampler \
            -labsl_raw_logging_internal -labsl_synchronization \
            -labsl_base -labsl_throw_delegate -labsl_strings

BUILD := build
BIN   := $(BUILD)/bench

.PHONY: all clean run sweep latency distribution

all: $(BIN)

$(BIN): benchmarks/bench.cpp src/pipeline.hpp src/fast_hash_set.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) benchmarks/bench.cpp -o $(BIN) $(LDFLAGS)

$(BUILD):
	mkdir -p $(BUILD)

run: $(BIN)
	$(BIN) --mode throughput --n 10000000 --k 1000 --dup 0.20

sweep: $(BIN)
	$(BIN) --mode sweep --n 10000000 --dup 0.20

latency: $(BIN)
	$(BIN) --mode latency --n 5000000 --k 1000 --dup 0.20

distribution: $(BIN)
	$(BIN) --mode distribution --n 5000000 --k 1000 --dup 0.20

clean:
	rm -rf $(BUILD)
