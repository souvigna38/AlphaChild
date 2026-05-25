# CPU-only educational build (DeepSeek-V4 skeleton + DeepSeek-V2 piece port)
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

BIN_V4 = bin/train_deepseek_v4_tiny
BIN_MLA = bin/test_mla
BIN_MOE = bin/test_moe
BIN_BLOCK = bin/test_block
SRCS_V4 = deepseek_v4_config.c rmsnorm.c train_deepseek_v4_tiny.c
SRCS_MLA = deepseek_v2/mla.c deepseek_v2/test_mla.c
SRCS_MOE = deepseek_v2/moe.c deepseek_v2/test_moe.c
SRCS_BLOCK = rmsnorm.c deepseek_v2/block.c deepseek_v2/mla.c deepseek_v2/moe.c deepseek_v2/test_block.c

.PHONY: all clean test test_mla test_moe test_block test_v2

all: $(BIN_V4) $(BIN_MLA) $(BIN_MOE) $(BIN_BLOCK)

$(BIN_V4): $(SRCS_V4)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4) $(LDFLAGS)

$(BIN_MLA): $(SRCS_MLA)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_MLA) $(LDFLAGS)

test: $(BIN_V4)
	./$(BIN_V4)

test_mla: $(BIN_MLA)
	./$(BIN_MLA)

$(BIN_MOE): $(SRCS_MOE)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_MOE) $(LDFLAGS)

test_moe: $(BIN_MOE)
	./$(BIN_MOE)

$(BIN_BLOCK): $(SRCS_BLOCK)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_BLOCK) $(LDFLAGS)

test_block: $(BIN_BLOCK)
	./$(BIN_BLOCK)

# Phase 2–3: MLA + MoE + one full block
test_v2: test_mla test_moe test_block

clean:
	rm -rf bin
