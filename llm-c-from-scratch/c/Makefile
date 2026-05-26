# CPU-only educational build (DeepSeek-V4 skeleton + DeepSeek-V2 piece port)
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

BIN_V4 = bin/train_deepseek_v4_tiny
BIN_SWIGLU = bin/test_swiglu
BIN_HASH_MOE = bin/test_hash_moe
BIN_SLIDING = bin/test_sliding_attn
BIN_MHC = bin/test_mhc
BIN_V4_ATTN = bin/test_v4_attention
BIN_V4_MOE = bin/test_v4_moe
BIN_V4_MODEL = bin/test_v4_model
BIN_V4_TRAIN_HEAD = bin/test_v4_train_head
BIN_V4_TRAIN_FINAL = bin/test_v4_train_final
BIN_HASH_MOE_TRAIN = bin/test_hash_moe_train
BIN_MLA = bin/test_mla
BIN_MOE = bin/test_moe
BIN_BLOCK = bin/test_block
BIN_TRAIN = bin/train_v2_tiny
V4_CORE = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c swiglu.c hash_moe.c hash_moe_train.c moe.c sliding_attn.c hca_compressor.c csa_compressor.c indexer.c mhc.c v4_attention.c v4_layer.c v4_model.c v4_train.c adamw.c
SRCS_V4 = $(V4_CORE) deepseek_v2/data.c train_deepseek_v4_tiny.c
SRCS_V4_TRAIN_HEAD = $(V4_CORE) test_v4_train_head.c
SRCS_V4_TRAIN_FINAL = $(V4_CORE) test_v4_train_final.c
SRCS_V4_MOE = deepseek_v4_config.c v4_ops.c swiglu.c hash_moe.c moe.c test_moe.c
SRCS_V4_MOE_BIN = $(SRCS_V4_MOE)
SRCS_V4_MODEL = $(V4_CORE) test_v4_model.c
SRCS_SWIGLU = v4_ops.c swiglu.c test_swiglu.c
SRCS_HASH_MOE = deepseek_v4_config.c v4_ops.c swiglu.c hash_moe.c test_hash_moe.c
SRCS_SLIDING = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c sliding_attn.c test_sliding_attn.c
SRCS_MHC = deepseek_v4_config.c v4_ops.c mhc.c test_mhc.c
SRCS_V4_ATTN = $(V4_CORE) test_v4_attention.c
SRCS_MLA = deepseek_v2/ops.c deepseek_v2/mla.c deepseek_v2/test_mla.c
SRCS_MOE = deepseek_v2/ops.c deepseek_v2/moe.c deepseek_v2/test_moe.c
SRCS_BLOCK = rmsnorm.c deepseek_v2/ops.c deepseek_v2/block.c deepseek_v2/mla.c deepseek_v2/moe.c deepseek_v2/test_block.c
SRCS_TRAIN = rmsnorm.c deepseek_v2/ops.c deepseek_v2/data.c deepseek_v2/adamw.c deepseek_v2/model.c deepseek_v2/block.c deepseek_v2/block_train.c deepseek_v2/mla.c deepseek_v2/moe.c deepseek_v2/moe_train.c deepseek_v2/train_v2_tiny.c

.PHONY: all clean test test_v4 test_mla test_moe test_block test_v2 test_train_v2

all: $(BIN_V4) $(BIN_MLA) $(BIN_MOE) $(BIN_BLOCK) $(BIN_TRAIN)

$(BIN_V4): $(SRCS_V4)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_V4) $(LDFLAGS)

$(BIN_MLA): $(SRCS_MLA)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_MLA) $(LDFLAGS)

test: $(BIN_V4)
	./$(BIN_V4)

SRCS_HASH_MOE_TRAIN = deepseek_v4_config.c v4_ops.c swiglu.c hash_moe.c hash_moe_train.c test_hash_moe_train.c

test_v4: $(BIN_V4) $(BIN_SWIGLU) $(BIN_HASH_MOE) $(BIN_HASH_MOE_TRAIN) $(BIN_SLIDING) $(BIN_MHC) $(BIN_V4_ATTN) $(BIN_V4_MOE) $(BIN_V4_MODEL) $(BIN_V4_TRAIN_HEAD) $(BIN_V4_TRAIN_FINAL)
	./$(BIN_V4)
	./$(BIN_SWIGLU)
	./$(BIN_HASH_MOE)
	./$(BIN_SLIDING)
	./$(BIN_MHC)
	./$(BIN_V4_ATTN)
	./$(BIN_V4_MOE)
	./$(BIN_V4_MODEL)
	./$(BIN_V4_TRAIN_HEAD)
	./$(BIN_V4_TRAIN_FINAL)
	./$(BIN_HASH_MOE_TRAIN)

$(BIN_HASH_MOE_TRAIN): $(SRCS_HASH_MOE_TRAIN)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_HASH_MOE_TRAIN) $(LDFLAGS)

$(BIN_V4_TRAIN_HEAD): $(SRCS_V4_TRAIN_HEAD)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_TRAIN_HEAD) $(LDFLAGS)

$(BIN_V4_TRAIN_FINAL): $(SRCS_V4_TRAIN_FINAL)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_TRAIN_FINAL) $(LDFLAGS)

$(BIN_V4_MOE): $(SRCS_V4_MOE_BIN)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_MOE_BIN) $(LDFLAGS)

$(BIN_V4_MODEL): $(SRCS_V4_MODEL)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_MODEL) $(LDFLAGS)

$(BIN_SLIDING): $(SRCS_SLIDING)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_SLIDING) $(LDFLAGS)

$(BIN_MHC): $(SRCS_MHC)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_MHC) $(LDFLAGS)

$(BIN_V4_ATTN): $(SRCS_V4_ATTN)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_ATTN) $(LDFLAGS)

$(BIN_SWIGLU): $(SRCS_SWIGLU)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_SWIGLU) $(LDFLAGS)

$(BIN_HASH_MOE): $(SRCS_HASH_MOE)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_HASH_MOE) $(LDFLAGS)

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

$(BIN_TRAIN): $(SRCS_TRAIN)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_TRAIN) $(LDFLAGS)

test_train_v2: $(BIN_TRAIN)
	./$(BIN_TRAIN)
	./$(BIN_TRAIN) -train-head 20
	./$(BIN_TRAIN) -train-1layer 15
	./$(BIN_TRAIN) -train-full 10
	./$(BIN_TRAIN) -train-adam 8 -batch 2

# Phase 2–4: pieces + trainer
test_v2: test_mla test_moe test_block test_train_v2

clean:
	rm -rf bin
