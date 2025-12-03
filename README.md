# Embedlet

A compact, single-header C library for fast embedding lookups and comparisons.

## Features

- **Single-header** - Just include `embedlet.h`
- **Fast SIMD** - Automatic AVX-512/AVX2/SSE optimization  
- **GPU Acceleration** - Transparent CUDA support (optional)
- **Thread-safe** - Multi-threaded similarity search
- **Memory-mapped** - Efficient large-scale storage
- **Cross-platform** - Windows, Linux, macOS

## Quick Start

### Build with CMake

**Windows (64-bit):**
```batch
cmake_build64.bat
cd sample_data
..\build\example.exe
```

**Linux/macOS:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd sample_data && ../build/example
```

### Basic Usage

```c
#define EMBEDLET_IMPLEMENTATION
#include "embedlet.h"

int main() {
    embedlet_store_t *store;
    embedlet_open("embeddings.emb", 1024, &store);

    float embedding[1024] = {...};
    size_t id;
    embedlet_append(store, embedding, false, &id);

    embedlet_result_t results[10];
    size_t count;
    embedlet_search(store, embedding, 10, true,
                   EMBEDLET_AUTO_THREADS, results, &count);

    embedlet_close(store, true);
}
```


## Performance

**500,000 embeddings × 1024 dimensions:**
(Intel i7-12700H, MinGW-w64 15.2.0, Windows 11 Pro)
- Top-5 search (20x average search time): Single-threaded: 0.176s
- Top-5 search (20x average search time): Multi-threaded: 0.096s

## Documentation

- **[docs/README.md](docs/README.md)** - API docs

## Requirements

- C11 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+
- Optional: SSE2-capable CPU

## License

Boost Software License 1.0.
See [LICENSE](LICENSE)
