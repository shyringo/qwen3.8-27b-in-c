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
LDFLAGS ?= -lm -pthread $(OMP_LDFLAGS)
INCLUDES := -Iinclude -Iinclude/qwen38 -Ithird_party -Isrc/io -Isrc/cli

GGUF_OBJ := $(BUILD)/src/io/qwen38_gguf.o
QUANT_OBJ := $(BUILD)/src/qwen38/qwen38_quant.o
MODEL_OBJ := $(BUILD)/src/qwen38/qwen38_model.o
TOKENIZER_OBJ := $(BUILD)/src/qwen38/qwen38_tokenizer.o
SAMPLER_OBJ := $(BUILD)/src/qwen38/qwen38_sampler.o
HTTP_OBJ := $(BUILD)/src/cli/qwen38_http.o
TOOL_OBJ := $(BUILD)/src/cli/qwen38_tool.o
TEST_BINS := $(BIN)/test_qwen38_gguf $(BIN)/test_qwen38_quant \
	$(BIN)/test_qwen38_sampler $(BIN)/test_qwen38_nfc \
	$(BIN)/test_qwen38_http $(BIN)/test_qwen38_tool
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

$(GGUF_OBJ): include/qwen38/qwen38_gguf.h
$(QUANT_OBJ): include/qwen38/qwen38_quant.h include/qwen38/qwen38_gguf.h \
	third_party/ggml-common.h
$(MODEL_OBJ): include/qwen38/qwen38_model.h include/qwen38/qwen38_quant.h \
	include/qwen38/qwen38_gguf.h
$(TOKENIZER_OBJ): include/qwen38/qwen38_tokenizer.h \
	include/qwen38/qwen38_gguf.h third_party/tok.h third_party/tok_unicode.h \
	third_party/tok_nfc.h third_party/tok_nfc_data.h
$(SAMPLER_OBJ): include/qwen38/qwen38_sampler.h

$(BIN):
	@mkdir -p $@

$(BIN)/test_qwen38_gguf: tests/unit/test_qwen38_gguf.c $(TOKENIZER_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_quant: tests/unit/test_qwen38_quant.c $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_sampler: tests/unit/test_qwen38_sampler.c $(SAMPLER_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_nfc: tests/unit/test_qwen38_nfc.c \
	third_party/tok_nfc.h third_party/tok_nfc_data.h | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_http: tests/unit/test_qwen38_http.c $(HTTP_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_tool: tests/unit/test_qwen38_tool.c $(TOOL_OBJ) $(HTTP_OBJ) | $(BIN)
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

$(BIN)/qwen38: src/cli/qwen38_main.c $(MODEL_OBJ) $(QUANT_OBJ) $(TOKENIZER_OBJ) $(SAMPLER_OBJ) $(GGUF_OBJ) $(HTTP_OBJ) $(TOOL_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

test: $(TEST_BINS) $(BIN)/qwen38
	./$(BIN)/test_qwen38_gguf
	./$(BIN)/test_qwen38_quant
	./$(BIN)/test_qwen38_sampler
	./$(BIN)/test_qwen38_nfc
	./$(BIN)/test_qwen38_http
	./$(BIN)/test_qwen38_tool
	./$(BIN)/qwen38 --help >/dev/null 2>&1

strict:
	$(MAKE) BUILD=build/strict BIN=build/strict/bin \
		CFLAGS="-O3 -std=c99 $(WARN) -Werror $(ARCH) $(OMP_CFLAGS) -ffp-contract=off" all tools test

portable:
	$(MAKE) BUILD=build/portable BIN=build/portable/bin \
		ARCH= OMP_CFLAGS= OMP_LDFLAGS= all tools test

clean:
	rm -rf $(BUILD) $(BIN)
