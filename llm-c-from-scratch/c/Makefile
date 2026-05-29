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
BIN_V4_TRAIN_1LAYER = bin/test_v4_train_1layer
BIN_V4_TRAIN_FULL = bin/test_v4_train_full
BIN_V4_TRAIN_4LAYER = bin/test_v4_train_4layer
BIN_V4_TRAIN_E2E = bin/test_v4_train_e2e
BIN_V4_PARITY = bin/test_v4_parity
BIN_V4_LAYER_PARITY = bin/test_v4_layer_parity
BIN_V4_TIED_HEAD = bin/test_v4_tied_head
BIN_INDEXER_TRAIN = bin/test_indexer_train
BIN_CUDA_RMSNORM = bin/test_cuda_rmsnorm
BIN_CUDA_SWIGLU = bin/test_cuda_swiglu
BIN_CUDA_CORE_ATTN = bin/test_cuda_core_attention
BIN_CUDA_SLIDING = bin/test_cuda_sliding_integrated
DS4_CUDA_CPU_SRCS = ds4_cuda.c rmsnorm.c swiglu.c rope.c sliding_attn.c v4_ops.c deepseek_v4_config.c
V4_LINK_CUDA = $(CUDA_OBJS)
ifeq ($(HAVE_NVCC),1)
  V4_CFLAGS_CUDA = $(DS4_CUDA_CFLAGS)
  V4_LIBS_CUDA = $(DS4_CUDA_LDFLAGS)
endif
NVCC ?= nvcc
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)
ifeq ($(HAVE_NVCC),1)
  CUDA_OBJS = cuda/ds4_rmsnorm.o cuda/ds4_swiglu.o cuda/ds4_core_attention.o
  DS4_CUDA_CFLAGS = -DDS4_HAVE_CUDA
  DS4_CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart
else
  CUDA_OBJS =
endif
BIN_HASH_MOE_TRAIN = bin/test_hash_moe_train
BIN_MLA = bin/test_mla
BIN_MOE = bin/test_moe
BIN_BLOCK = bin/test_block
BIN_TRAIN = bin/train_v2_tiny
V4_CORE = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c swiglu.c hash_moe.c hash_moe_train.c moe.c sliding_attn.c sliding_attn_train.c hca_compressor.c hca_compressor_train.c csa_compressor.c csa_compressor_train.c indexer.c indexer_train.c mhc.c v4_attention.c v4_attention_train.c v4_layer.c v4_layer_train.c v4_model.c v4_train.c adamw.c ds4_cuda.c
SRCS_V4 = $(V4_CORE) deepseek_v2/data.c train_deepseek_v4_tiny.c
SRCS_V4_TRAIN_HEAD = $(V4_CORE) test_v4_train_head.c
SRCS_V4_TRAIN_FINAL = $(V4_CORE) test_v4_train_final.c
SRCS_V4_TRAIN_1LAYER = $(V4_CORE) test_v4_train_1layer.c
SRCS_V4_TRAIN_FULL = $(V4_CORE) test_v4_train_full.c
SRCS_V4_TRAIN_4LAYER = $(V4_CORE) test_v4_train_4layer.c
SRCS_V4_TRAIN_E2E = $(V4_CORE) test_v4_train_e2e.c
SRCS_V4_PARITY = $(V4_CORE) v4_parity.c test_v4_parity.c
SRCS_V4_LAYER_PARITY = $(V4_CORE) v4_parity.c test_v4_layer_parity.c
SRCS_V4_TIED_HEAD = $(V4_CORE) test_v4_tied_head.c
SRCS_INDEXER_TRAIN = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c indexer.c indexer_train.c test_indexer_train.c
SRCS_V4_MODEL = $(V4_CORE) test_v4_model.c
SRCS_SWIGLU = v4_ops.c swiglu.c test_swiglu.c
SRCS_HASH_MOE = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c swiglu.c sliding_attn.c ds4_cuda.c hash_moe.c test_hash_moe.c
SRCS_V4_MOE = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c swiglu.c sliding_attn.c ds4_cuda.c hash_moe.c moe.c test_moe.c
SRCS_V4_MOE_BIN = $(SRCS_V4_MOE)
SRCS_SLIDING = deepseek_v4_config.c rmsnorm.c swiglu.c v4_ops.c rope.c sliding_attn.c ds4_cuda.c test_sliding_attn.c
SRCS_MHC = deepseek_v4_config.c v4_ops.c mhc.c test_mhc.c
SRCS_V4_ATTN = $(V4_CORE) test_v4_attention.c
SRCS_MLA = deepseek_v2/ops.c deepseek_v2/mla.c deepseek_v2/test_mla.c
SRCS_MOE = deepseek_v2/ops.c deepseek_v2/moe.c deepseek_v2/test_moe.c
SRCS_BLOCK = rmsnorm.c deepseek_v2/ops.c deepseek_v2/block.c deepseek_v2/mla.c deepseek_v2/moe.c deepseek_v2/test_block.c
SRCS_TRAIN = rmsnorm.c deepseek_v2/ops.c deepseek_v2/data.c deepseek_v2/adamw.c deepseek_v2/model.c deepseek_v2/block.c deepseek_v2/block_train.c deepseek_v2/mla.c deepseek_v2/moe.c deepseek_v2/moe_train.c deepseek_v2/train_v2_tiny.c

.PHONY: all clean test test_v4 test_mla test_moe test_block test_v2 test_train_v2 cuda-stub

all: $(BIN_V4) $(BIN_MLA) $(BIN_MOE) $(BIN_BLOCK) $(BIN_TRAIN)

$(BIN_V4): $(SRCS_V4) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -Ideepseek_v2 -o $@ $(SRCS_V4) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_MLA): $(SRCS_MLA)
	mkdir -p bin
	$(CC) $(CFLAGS) -Ideepseek_v2 -o $@ $(SRCS_MLA) $(LDFLAGS)

test: $(BIN_V4)
	./$(BIN_V4)

SRCS_HASH_MOE_TRAIN = deepseek_v4_config.c rmsnorm.c v4_ops.c rope.c swiglu.c sliding_attn.c ds4_cuda.c hash_moe.c hash_moe_train.c test_hash_moe_train.c

test_v4: $(BIN_V4) $(BIN_SWIGLU) $(BIN_HASH_MOE) $(BIN_HASH_MOE_TRAIN) $(BIN_SLIDING) $(BIN_MHC) $(BIN_V4_ATTN) $(BIN_V4_MOE) $(BIN_V4_MODEL) $(BIN_V4_TRAIN_HEAD) $(BIN_V4_TRAIN_FINAL) $(BIN_V4_TRAIN_1LAYER) $(BIN_V4_TRAIN_FULL) $(BIN_V4_TRAIN_4LAYER) $(BIN_V4_TRAIN_E2E) $(BIN_V4_PARITY) $(BIN_V4_LAYER_PARITY) $(BIN_V4_TIED_HEAD) $(BIN_CUDA_RMSNORM) $(BIN_CUDA_SWIGLU) $(BIN_CUDA_CORE_ATTN) $(BIN_CUDA_SLIDING) $(BIN_INDEXER_TRAIN)
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
	./$(BIN_V4_TRAIN_1LAYER)
	./$(BIN_V4_TRAIN_FULL)
	./$(BIN_V4_TRAIN_4LAYER)
	./$(BIN_V4_TRAIN_E2E)
	./$(BIN_V4_PARITY)
	./$(BIN_V4_LAYER_PARITY)
	./$(BIN_V4_TIED_HEAD)
	./$(BIN_CUDA_RMSNORM)
	./$(BIN_CUDA_SWIGLU)
	./$(BIN_CUDA_CORE_ATTN)
	./$(BIN_CUDA_SLIDING)
	./$(BIN_INDEXER_TRAIN)
	./$(BIN_HASH_MOE_TRAIN)

$(BIN_HASH_MOE_TRAIN): $(SRCS_HASH_MOE_TRAIN)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_HASH_MOE_TRAIN) $(LDFLAGS)

$(BIN_V4_TRAIN_HEAD): $(SRCS_V4_TRAIN_HEAD) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_TRAIN_HEAD) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_TRAIN_FINAL): $(SRCS_V4_TRAIN_FINAL) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_TRAIN_FINAL) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_TRAIN_1LAYER): $(SRCS_V4_TRAIN_1LAYER) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_TRAIN_1LAYER) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_TRAIN_FULL): $(SRCS_V4_TRAIN_FULL) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_TRAIN_FULL) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_TRAIN_4LAYER): $(SRCS_V4_TRAIN_4LAYER) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_TRAIN_4LAYER) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_TRAIN_E2E): $(SRCS_V4_TRAIN_E2E) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_TRAIN_E2E) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_PARITY): $(SRCS_V4_PARITY) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_PARITY) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_LAYER_PARITY): $(SRCS_V4_LAYER_PARITY) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_LAYER_PARITY) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_V4_TIED_HEAD): $(SRCS_V4_TIED_HEAD)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_TIED_HEAD) $(LDFLAGS)

$(BIN_INDEXER_TRAIN): $(SRCS_INDEXER_TRAIN)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_INDEXER_TRAIN) $(LDFLAGS)

ifeq ($(HAVE_NVCC),1)
cuda/ds4_rmsnorm.o: cuda/ds4_rmsnorm.cu
	$(NVCC) -O2 -std=c++14 -c -o $@ $<
cuda/ds4_swiglu.o: cuda/ds4_swiglu.cu
	$(NVCC) -O2 -std=c++14 -c -o $@ $<
cuda/ds4_core_attention.o: cuda/ds4_core_attention.cu
	$(NVCC) -O2 -std=c++14 -c -o $@ $<
endif

$(BIN_CUDA_RMSNORM): $(DS4_CUDA_CPU_SRCS) test_cuda_rmsnorm.c $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(DS4_CUDA_CFLAGS) -o $@ $(DS4_CUDA_CPU_SRCS) test_cuda_rmsnorm.c $(CUDA_OBJS) $(LDFLAGS) $(DS4_CUDA_LDFLAGS)

$(BIN_CUDA_SWIGLU): $(DS4_CUDA_CPU_SRCS) test_cuda_swiglu.c $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(DS4_CUDA_CFLAGS) -o $@ $(DS4_CUDA_CPU_SRCS) test_cuda_swiglu.c $(CUDA_OBJS) $(LDFLAGS) $(DS4_CUDA_LDFLAGS)

$(BIN_CUDA_CORE_ATTN): $(DS4_CUDA_CPU_SRCS) test_cuda_core_attention.c $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(DS4_CUDA_CFLAGS) -o $@ $(DS4_CUDA_CPU_SRCS) test_cuda_core_attention.c $(CUDA_OBJS) $(LDFLAGS) $(DS4_CUDA_LDFLAGS)

$(BIN_CUDA_SLIDING): $(DS4_CUDA_CPU_SRCS) test_cuda_sliding_integrated.c $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(DS4_CUDA_CFLAGS) -o $@ $(DS4_CUDA_CPU_SRCS) test_cuda_sliding_integrated.c $(CUDA_OBJS) $(LDFLAGS) $(DS4_CUDA_LDFLAGS)

cuda-stub: $(BIN_CUDA_RMSNORM) $(BIN_CUDA_SWIGLU) $(BIN_CUDA_CORE_ATTN) $(BIN_CUDA_SLIDING)
	./$(BIN_CUDA_RMSNORM)
	./$(BIN_CUDA_SWIGLU)
	./$(BIN_CUDA_CORE_ATTN)
	./$(BIN_CUDA_SLIDING)

$(BIN_V4_MOE): $(SRCS_V4_MOE_BIN)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_V4_MOE_BIN) $(LDFLAGS)

$(BIN_V4_MODEL): $(SRCS_V4_MODEL) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_MODEL) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

$(BIN_SLIDING): $(SRCS_SLIDING)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_SLIDING) $(LDFLAGS)

$(BIN_MHC): $(SRCS_MHC)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS_MHC) $(LDFLAGS)

$(BIN_V4_ATTN): $(SRCS_V4_ATTN) $(CUDA_OBJS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(V4_CFLAGS_CUDA) -o $@ $(SRCS_V4_ATTN) $(V4_LINK_CUDA) $(LDFLAGS) $(V4_LIBS_CUDA)

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
