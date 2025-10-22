cmake_minimum_required(VERSION 3.10)
project(csl-sys-cuda LANGUAGES CXX)

# Set the project root directory
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Pass through CUDA architectures if specified
if(DEFINED CUDA_ARCHS)
    set(ENV{CUDA_ARCHS} "${CUDA_ARCHS}")
endif()

# Pass through CUDA version if specified
if(DEFINED CUDA_VERSION)
    set(ENV{CUDA_VERSION} "${CUDA_VERSION}")
endif()

# Pass through profile debug flag if specified
if(DEFINED PROFILE_DEBUG_DATA)
    set(ENV{PROFILE_DEBUG_DATA} "${PROFILE_DEBUG_DATA}")
endif()

# Determine build type for Make
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MAKE_BUILD_TYPE "Debug")
else()
    set(MAKE_BUILD_TYPE "Release")
endif()

# Pass cbindgen include directory if specified
if(DEFINED CBINDGEN_INCLUDE_DIR)
    set(CBINDGEN_FLAG "CBINDGEN_INCLUDE_DIR=${CBINDGEN_INCLUDE_DIR}")
endif()

# Custom target to invoke Makefile
# USES_TERMINAL gives the command access to the console pool, which helps
# with jobserver propagation when building under cargo's parallel execution
add_custom_target(cuda_library ALL
    COMMAND ${CMAKE_MAKE_PROGRAM}
            -C ${PROJECT_ROOT}
            BUILD_TYPE=${MAKE_BUILD_TYPE}
            ${CBINDGEN_FLAG}
            all
    COMMENT "Building CUDA library via Makefile"
    USES_TERMINAL
    VERBATIM
)

# Create a dummy library target for CMake to track
add_library(sys-cuda STATIC IMPORTED GLOBAL)

# Set the location of the built library
set_target_properties(sys-cuda PROPERTIES
    IMPORTED_LOCATION "${PROJECT_ROOT}/target/cuda-build/lib/libsys-cuda.a"
)

# Add dependency on the custom target
add_dependencies(sys-cuda cuda_library)

# Install rule
install(FILES "${PROJECT_ROOT}/target/cuda-build/lib/libsys-cuda.a"
        DESTINATION lib)

# Export the library location for the cmake crate
set(CSL_SYS_CUDA_LIB "${PROJECT_ROOT}/target/cuda-build/lib/libsys-cuda.a" CACHE FILEPATH "Path to the CUDA library")

# Clean target
add_custom_target(clean_cuda
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${PROJECT_ROOT} clean
    COMMENT "Cleaning CUDA build artifacts"
    USES_TERMINAL
    VERBATIM
)