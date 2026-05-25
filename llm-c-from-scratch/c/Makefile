# CPU-only educational build (DeepSeek-V4 skeleton + DeepSeek-V2 piece port)
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

BIN_V4 = bin/train_deepseek_v4_tiny
BIN_MLA = bin/test_mla
SRCS_V4 = deepseek_v4_config.c rmsnorm.c train_deepseek_v4_tiny.c
SRCS_MLA = deepseek_v2/mla.c deepseek_v2/test_mla.c

.PHONY: all clean test test_mla

all: $(BIN_V4) $(BIN_MLA)

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

clean:
	rm -rf bin
