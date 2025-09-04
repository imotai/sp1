# CUDA Compilation Makefile for CSL-SYS
# This Makefile compiles all CUDA sources from cuda/ and sppark/ directories

# Detect NVCC
NVCC := $(shell which nvcc)
ifndef NVCC
$(error nvcc not found. Please ensure CUDA toolkit is installed)
endif

# Get CUDA version
CUDA_VERSION := $(shell nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
CUDA_VERSION_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d. -f1)
CUDA_VERSION_MINOR := $(shell echo $(CUDA_VERSION) | cut -d. -f2)

# Check minimum CUDA version (12.0)
ifeq ($(shell echo "$(CUDA_VERSION_MAJOR) < 12" | bc), 1)
$(error CUDA version $(CUDA_VERSION) is not supported. Minimum required: 12.0)
endif

# Directories
PROJECT_ROOT := $(shell pwd)
CUDA_DIR := $(PROJECT_ROOT)/cuda
SPPARK_DIR := $(PROJECT_ROOT)/sppark
BUILD_DIR := $(PROJECT_ROOT)/target/cuda-build
OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := $(BUILD_DIR)/lib
INCLUDE_DIR := $(BUILD_DIR)/include

# Output library name
LIB_NAME := libsys-cuda.a

# Compiler flags
NVCC_FLAGS := -std=c++20 \
              -default-stream=per-thread \
              -Xcompiler -fopenmp \
			  -Xcompiler -fPIE \
              -lnvToolsExt \
              -lineinfo \
              -ldl \
              --expt-relaxed-constexpr \
              --use_fast_math \
              -I$(CUDA_DIR) \
              -I$(SPPARK_DIR) \
              -DSPPARK \
              -DFEATURE_KOALA_BEAR

# Add cbindgen include directory if provided
ifdef CBINDGEN_INCLUDE_DIR
    NVCC_FLAGS += -I$(CBINDGEN_INCLUDE_DIR)
endif

# Architecture flags
ifdef CUDA_ARCHS
    ARCH_FLAGS := $(foreach arch,$(subst $(comma), ,$(CUDA_ARCHS)),-gencode=arch=compute_$(arch),code=sm_$(arch))
    # Find max architecture for PTX fallback
    MAX_ARCH := $(lastword $(sort $(subst $(comma), ,$(CUDA_ARCHS))))
    ARCH_FLAGS += -gencode=arch=compute_$(MAX_ARCH),code=compute_$(MAX_ARCH)
else
    # Default architectures based on CUDA version.
    ifeq ($(shell echo "$(CUDA_VERSION_MAJOR) == 12 && $(CUDA_VERSION_MINOR) >= 8" | bc), 1)
        ARCH_FLAGS := -gencode=arch=compute_80,code=sm_80 \
                      -gencode=arch=compute_86,code=sm_86 \
                      -gencode=arch=compute_89,code=sm_89 \
                      -gencode=arch=compute_90,code=sm_90 \
                      -gencode=arch=compute_100,code=sm_100 \
                      -gencode=arch=compute_120,code=sm_120 \
                      -gencode=arch=compute_120,code=compute_120
    else ifeq ($(shell echo "$(CUDA_VERSION_MAJOR) == 13" | bc), 1)
        ARCH_FLAGS := -gencode=arch=compute_80,code=sm_80 \
                      -gencode=arch=compute_86,code=sm_86 \
                      -gencode=arch=compute_89,code=sm_89 \
                      -gencode=arch=compute_90,code=sm_90 \
                      -gencode=arch=compute_100,code=sm_100 \
                      -gencode=arch=compute_130,code=sm_130 \
                      -gencode=arch=compute_130,code=compute_130
    else
        ARCH_FLAGS := -gencode=arch=compute_80,code=sm_80 \
                      -gencode=arch=compute_86,code=sm_86 \
                      -gencode=arch=compute_89,code=sm_89 \
                      -gencode=arch=compute_89,code=compute_89
    endif
endif

# Build configuration
BUILD_TYPE ?= Release

# Debug/Release specific flags
#
# For now, they both use the same profile as it seems to cause some issues with one of the CUDA
# sources. In the future, we should use different profiles for each.
ifeq ($(BUILD_TYPE),Debug)
    NVCC_FLAGS += -O3
    BUILD_SUFFIX := release
else
    NVCC_FLAGS += -O3
    BUILD_SUFFIX := release
endif

# Profile debug data flag
ifdef PROFILE_DEBUG_DATA
ifeq ($(PROFILE_DEBUG_DATA),true)
    NVCC_FLAGS += -G
endif
endif

# Find all source files
CUDA_SOURCES := $(shell find $(CUDA_DIR) -name "*.cu")
SPPARK_SOURCES := $(SPPARK_DIR)/rust/src/lib.cpp $(SPPARK_DIR)/util/all_gpus.cpp

# Generate object file names
CUDA_OBJECTS := $(patsubst $(CUDA_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CUDA_SOURCES))
SPPARK_OBJECTS := $(OBJ_DIR)/sppark/lib.o $(OBJ_DIR)/sppark/all_gpus.o

# All objects
ALL_OBJECTS := $(CUDA_OBJECTS) $(SPPARK_OBJECTS)

# Create directory structure for object files
CUDA_OBJ_DIRS := $(sort $(dir $(CUDA_OBJECTS)))
SPPARK_OBJ_DIRS := $(sort $(dir $(SPPARK_OBJECTS)))
ALL_OBJ_DIRS := $(CUDA_OBJ_DIRS) $(SPPARK_OBJ_DIRS)

# Default target
.PHONY: all
all: $(LIB_DIR)/$(LIB_NAME)

# Create directories
$(ALL_OBJ_DIRS) $(LIB_DIR) $(INCLUDE_DIR):
	@mkdir -p $@

# Main library target - create library first, then device-link
$(LIB_DIR)/$(LIB_NAME): $(ALL_OBJECTS) | $(LIB_DIR) $(OBJ_DIR)
	@echo "Creating initial static library $@"
	@ar rcs $@ $^
	@echo "Device linking library..."
	@$(NVCC) $(ARCH_FLAGS) --device-link -o $(OBJ_DIR)/device_link.o $@
	@echo "Adding device link to library..."
	@ar r $@ $(OBJ_DIR)/device_link.o
	@echo "Library created: $@"

# CUDA compilation rules
$(OBJ_DIR)/cuda/%.o: $(CUDA_DIR)/%.cu | $(CUDA_OBJ_DIRS)
	@echo "Compiling CUDA: $<"
	@$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c $< -o $@

# Sppark C++ compilation rules  
$(OBJ_DIR)/sppark/lib.o: $(SPPARK_DIR)/rust/src/lib.cpp | $(SPPARK_OBJ_DIRS)
	@echo "Compiling C++: $<"
	@$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c $< -o $@

$(OBJ_DIR)/sppark/all_gpus.o: $(SPPARK_DIR)/util/all_gpus.cpp | $(SPPARK_OBJ_DIRS)
	@echo "Compiling C++: $<"
	@$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c $< -o $@

# Module-specific targets for convenience
.PHONY: algebra basefold challenger jagged logup_gkr merkle mle ntt prover-clean reduce runtime scan sumcheck tracegen transpose zerocheck

algebra: $(filter $(OBJ_DIR)/cuda/algebra/%,$(CUDA_OBJECTS))
basefold: $(filter $(OBJ_DIR)/cuda/basefold/%,$(CUDA_OBJECTS))
challenger: $(filter $(OBJ_DIR)/cuda/challenger/%,$(CUDA_OBJECTS))
jagged: $(filter $(OBJ_DIR)/cuda/jagged/%,$(CUDA_OBJECTS))
logup_gkr: $(filter $(OBJ_DIR)/cuda/logup_gkr/%,$(CUDA_OBJECTS))
merkle: $(filter $(OBJ_DIR)/cuda/merkle-tree/%,$(CUDA_OBJECTS))
mle: $(filter $(OBJ_DIR)/cuda/mle/%,$(CUDA_OBJECTS))
ntt: $(filter $(OBJ_DIR)/cuda/ntt/%,$(CUDA_OBJECTS))
prover-clean: $(filter $(OBJ_DIR)/cuda/prover-clean/%,$(CUDA_OBJECTS))
reduce: $(filter $(OBJ_DIR)/cuda/reduce/%,$(CUDA_OBJECTS))
runtime: $(filter $(OBJ_DIR)/cuda/runtime/%,$(CUDA_OBJECTS))
scan: $(filter $(OBJ_DIR)/cuda/scan/%,$(CUDA_OBJECTS))
sumcheck: $(filter $(OBJ_DIR)/cuda/sumcheck/%,$(CUDA_OBJECTS))
tracegen: $(filter $(OBJ_DIR)/cuda/tracegen/%,$(CUDA_OBJECTS))
transpose: $(filter $(OBJ_DIR)/cuda/transpose/%,$(CUDA_OBJECTS))
zerocheck: $(filter $(OBJ_DIR)/cuda/zerocheck/%,$(CUDA_OBJECTS))

# Build type targets
.PHONY: debug release

debug:
	@$(MAKE) BUILD_TYPE=Debug

release:
	@$(MAKE) BUILD_TYPE=Release

# Clean target
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)

# Install target (for CMake integration)
.PHONY: install
install: $(LIB_DIR)/$(LIB_NAME)
	@echo "Installing library to $(DESTDIR)"
	@mkdir -p $(DESTDIR)/lib
	@cp $(LIB_DIR)/$(LIB_NAME) $(DESTDIR)/lib/

# Help target
.PHONY: help
help:
	@echo "CSL-SYS CUDA Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make [target] [BUILD_TYPE=Debug|Release] [CUDA_ARCHS=80,86,89]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build all CUDA sources (default)"
	@echo "  debug     - Build with debug flags"
	@echo "  release   - Build with optimization"
	@echo "  clean     - Remove all build artifacts"
	@echo "  install   - Install library to DESTDIR"
	@echo ""
	@echo "Module targets:"
	@echo "  algebra, basefold, challenger, jagged, logup_gkr, merkle,"
	@echo "  mle, ntt, prover-clean, reduce, runtime, scan, sumcheck,"
	@echo "  tracegen, transpose, zerocheck"
	@echo ""
	@echo "Environment variables:"
	@echo "  BUILD_TYPE        - Debug or Release (default: Release)"
	@echo "  CUDA_ARCHS        - Comma-separated GPU architectures (default: auto-detect)"
	@echo "  PROFILE_DEBUG_DATA - Set to 'true' to enable -G flag"
	@echo "  DESTDIR           - Installation directory for 'make install'"

# Print configuration
.PHONY: info
info:
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "Build Type: $(BUILD_TYPE)"
	@echo "Architecture Flags: $(ARCH_FLAGS)"
	@echo "Build Directory: $(BUILD_DIR)"
	@echo "Library Output: $(LIB_DIR)/$(LIB_NAME)"

# Dependency tracking
-include $(CUDA_OBJECTS:.o=.d)
-include $(SPPARK_OBJECTS:.o=.d)