CXX    ?= g++
BUILD  ?= dev

COMMON_FLAGS := -std=c++14 -march=native -isystem ./third_party/Eigen
COMMON_FLAGS += -Wall -Wextra

HARDENING_FLAGS := -D_FORTIFY_SOURCE=2 -D_GLIBCXX_ASSERTIONS
HARDENING_FLAGS += -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST

ifeq ($(BUILD),release)
    CXXFLAGS ?= -O2 -DNDEBUG -DEIGEN_NO_DEBUG
    CXXFLAGS += $(COMMON_FLAGS) $(HARDENING_FLAGS)
else ifeq ($(BUILD),dev)
    CXXFLAGS ?= -O0 -g
    CXXFLAGS += $(COMMON_FLAGS) $(HARDENING_FLAGS)
    CXXFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer
    LDFLAGS  += -fsanitize=address,undefined
else
    $(error Unknown BUILD type: $(BUILD). Use 'dev' or 'release')
endif

OUT_DIR := out
TARGET  := $(OUT_DIR)/fracture

.PHONY: all clean lint-python

all: $(TARGET)

$(TARGET): fracture.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)
	rm -f graph*.dot* *.csv
	rm -rf data graphs pdf results

lint-python:
	uv run ruff format
	uv run ruff check
	uv run pyrefly check
