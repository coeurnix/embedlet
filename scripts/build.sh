#!/bin/bash
# scripts/build.sh
# Build script for Linux/MacOS (GCC/Clang)
# Usage: ./build.sh [gcc|clang] [debug|release] [avx512|avx2|sse|native|generic]

set -e

# Change to project root
cd "$(dirname "$0")/.."

COMPILER="${1:-gcc}"
CONFIG="${2:-release}"
SIMD="${3:-native}"

INCLUDE_DIR="include"
EXAMPLES_DIR="examples"
TESTS_DIR="tests"
BUILD_DIR="build/${COMPILER}/${CONFIG}"

mkdir -p "$BUILD_DIR"

echo ""
echo "============================================"
echo " Building embedlet ($COMPILER / $CONFIG / $SIMD)"
echo "============================================"
echo ""

# Select compiler
case "$COMPILER" in
    gcc)
        CC=gcc
        ;;
    clang)
        CC=clang
        ;;
    *)
        echo "Unknown compiler: $COMPILER"
        echo "Usage: ./build.sh [gcc|clang] [debug|release] [avx512|avx2|sse|native|generic]"
        exit 1
        ;;
esac

# Check compiler exists
if ! command -v "$CC" &> /dev/null; then
    echo "Error: $CC not found"
    exit 1
fi

# Common flags
COMMON_FLAGS="-Wall -Wextra -Wpedantic -std=c11 -I${INCLUDE_DIR}"

# Optimization flags
case "$CONFIG" in
    debug)
        OPT_FLAGS="-O0 -g -DDEBUG"
        ;;
    release)
        OPT_FLAGS="-O3 -DNDEBUG"
        ;;
    *)
        echo "Unknown config: $CONFIG (use debug or release)"
        exit 1
        ;;
esac

# SIMD flags
case "$SIMD" in
    avx512)
        SIMD_FLAGS="-mavx512f -mavx512dq -mfma"
        ;;
    avx2)
        SIMD_FLAGS="-mavx2 -mfma"
        ;;
    sse)
        SIMD_FLAGS="-msse4.2 -msse4.1 -mssse3 -msse2"
        ;;
    native)
        SIMD_FLAGS="-march=native"
        ;;
    generic)
        SIMD_FLAGS=""
        ;;
    *)
        echo "Unknown SIMD option: $SIMD"
        echo "Options: avx512, avx2, sse, native, generic"
        exit 1
        ;;
esac

# Libraries
LIBS="-lm -lpthread"

# macOS doesn't need -lpthread explicitly
if [[ "$(uname)" == "Darwin" ]]; then
    LIBS="-lm"
fi

echo "Compiler: $CC"
echo "Flags: $COMMON_FLAGS $OPT_FLAGS $SIMD_FLAGS"
echo "Libraries: $LIBS"
echo ""

# Build test executable
echo "Building test_embedlet..."
$CC $COMMON_FLAGS $OPT_FLAGS $SIMD_FLAGS \
    -o "${BUILD_DIR}/test_embedlet" \
    "${TESTS_DIR}/test_embedlet.c" \
    $LIBS

# Build example executable  
echo "Building example..."
$CC $COMMON_FLAGS $OPT_FLAGS $SIMD_FLAGS \
    -o "${BUILD_DIR}/example" \
    "${EXAMPLES_DIR}/example.c" \
    $LIBS

echo ""
echo "============================================"
echo " Build complete!"
echo "============================================"
echo ""
echo "Executables in: ${BUILD_DIR}/"
echo ""
echo "To run tests:"
echo "  cd sample_data"
echo "  ../${BUILD_DIR}/test_embedlet"
echo ""
echo "To run example:"
echo "  cd sample_data"
echo "  ../${BUILD_DIR}/example"
echo ""