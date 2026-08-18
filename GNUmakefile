CC       ?= cc
BUILD    ?= build
BIN      ?= bin

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH ?= -mcpu=native
  else
    ARCH ?= -march=native
  endif
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp
else
  ARCH ?= -march=native
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

WARN := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla
CFLAGS ?= -O3 -std=c99 $(WARN) $(ARCH) $(OMP_CFLAGS) -ffp-contract=off
LDFLAGS ?= -lm $(OMP_LDFLAGS)
INCLUDES := -Iinclude -Iinclude/qwen38 -Ithird_party -Isrc/io

GGUF_OBJ := $(BUILD)/src/io/qwen38_gguf.o
QUANT_OBJ := $(BUILD)/src/qwen38/qwen38_quant.o
MODEL_OBJ := $(BUILD)/src/qwen38/qwen38_model.o
TOKENIZER_OBJ := $(BUILD)/src/qwen38/qwen38_tokenizer.o
SAMPLER_OBJ := $(BUILD)/src/qwen38/qwen38_sampler.o
TEST_BINS := $(BIN)/test_qwen38_gguf $(BIN)/test_qwen38_quant \
	$(BIN)/test_qwen38_sampler
TOOL_BINS := $(BIN)/qwen38-gguf-inspect \
	$(BIN)/qwen38-kernel-probe $(BIN)/qwen38-forward-probe \
	$(BIN)/qwen38-logits-probe $(BIN)/qwen38-batch-probe \
	$(BIN)/qwen38-mtp-probe $(BIN)/qwen38-spec-probe \
	$(BIN)/qwen38-spec-bench $(BIN)/qwen38-tokenize

.PHONY: all test tools strict portable clean

all: $(BIN)/qwen38

tools: $(TOOL_BINS)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BIN):
	@mkdir -p $@

$(BIN)/test_qwen38_gguf: tests/unit/test_qwen38_gguf.c $(TOKENIZER_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_quant: tests/unit/test_qwen38_quant.c $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_sampler: tests/unit/test_qwen38_sampler.c $(SAMPLER_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-gguf-inspect: src/cli/qwen38_gguf_inspect.c $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-kernel-probe: src/cli/qwen38_kernel_probe.c $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-forward-probe: src/cli/qwen38_forward_probe.c $(MODEL_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-logits-probe: src/cli/qwen38_logits_probe.c $(MODEL_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-batch-probe: src/cli/qwen38_batch_probe.c $(MODEL_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-mtp-probe: src/cli/qwen38_mtp_probe.c $(MODEL_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-spec-probe: src/cli/qwen38_spec_probe.c $(MODEL_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-spec-bench: src/cli/qwen38_spec_bench.c $(MODEL_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38-tokenize: src/cli/qwen38_tokenize.c $(TOKENIZER_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen38: src/cli/qwen38_main.c $(MODEL_OBJ) $(QUANT_OBJ) $(TOKENIZER_OBJ) $(SAMPLER_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

test: $(TEST_BINS) $(BIN)/qwen38
	./$(BIN)/test_qwen38_gguf
	./$(BIN)/test_qwen38_quant
	./$(BIN)/test_qwen38_sampler
	./$(BIN)/qwen38 --help >/dev/null 2>&1

strict:
	$(MAKE) BUILD=build/strict BIN=build/strict/bin \
		CFLAGS="-O3 -std=c99 $(WARN) -Werror $(ARCH) $(OMP_CFLAGS) -ffp-contract=off" all tools test

portable:
	$(MAKE) BUILD=build/portable BIN=build/portable/bin \
		ARCH= OMP_CFLAGS= OMP_LDFLAGS= all tools test

clean:
	rm -rf $(BUILD) $(BIN)
