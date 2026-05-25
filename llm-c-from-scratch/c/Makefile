# CPU-only educational build (DeepSeek-V4 tiny skeleton)
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

BIN = bin/train_deepseek_v4_tiny
SRCS = deepseek_v4_config.c rmsnorm.c train_deepseek_v4_tiny.c

.PHONY: all clean test

all: $(BIN)

$(BIN): $(SRCS)
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

test: all
	./$(BIN)

clean:
	rm -rf bin
