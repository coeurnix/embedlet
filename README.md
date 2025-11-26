# embedlet

**A tiny, single-header C library for storing and searching vector embeddings on disk.**

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

embedlet lets you store millions of vector embeddings in a memory-mapped file and search for the most (or least) similar ones using cosine similarity. It's designed to be simple, fast, and dependency-free.

---

## What Are Embeddings?

Modern AI models can convert almost anything—text, images, audio, code—into **embeddings**: arrays of floating-point numbers (typically 256 to 4096 dimensions) that capture semantic meaning.

The key is that **similar things have similar embeddings**. Two sentences with the same meaning will have embeddings that are mathematically close together, even if they use completely different words.

This enables powerful applications:

| Use Case | How It Works |
|----------|--------------|
| **Semantic Search** | Find documents by meaning, not just keywords |
| **Recommendations** | "Users who liked X also liked Y" |
| **Duplicate Detection** | Find near-identical content |
| **Clustering** | Group similar items automatically |
| **RAG (Retrieval-Augmented Generation)** | Give LLMs access to your private data |
| **Image Search** | Find visually similar images |
| **Anomaly Detection** | Find outliers that don't match the pattern |

embedlet handles the storage and similarity search part—you bring the embeddings from your favorite model (Qwen, OpenAI, Gemini, Voyage, etc.).

---

## Features

- **Single-header library** — just copy `embedlet.h` into your project
- **Memory-mapped storage** — handles datasets larger than RAM efficiently
- **SIMD-accelerated similarity** — AVX-512, AVX2, SSE, with pure C fallback
- **Multithreaded search** — optional thread pool for parallel queries
- **Cross-platform** — Windows and Linux with minimal platform code
- **Zero dependencies** — only the C standard library
- **Simple API** — feels natural to C developers

---

## Quick Start

### 1. Get the Header

Copy `include/embedlet.h` into your project, or clone the repository:

```bash
git clone https://github.com/coeurnix/embedlet.git
```

### 2. Write Some Code

```c
#define EMBEDLET_IMPLEMENTATION  // Only in ONE .c file
#include "embedlet.h"
#include <stdio.h>

int main(void) {
    embedlet_store_t *store;
    
    // Create a store for 384-dimensional embeddings
    embedlet_open("my_embeddings.db", 384, &store);
    
    // Add some embeddings (you'd get these from an ML model)
    float embedding1[384] = { /* ... */ };
    float embedding2[384] = { /* ... */ };
    
    size_t id1, id2;
    embedlet_append(store, embedding1, false, &id1);  // id1 = 0
    embedlet_append(store, embedding2, false, &id2);  // id2 = 1
    
    // Search for the 5 most similar to a query
    float query[384] = { /* ... */ };
    embedlet_result_t results[5];
    size_t count;
    
    embedlet_search(store, query, 5, true, EMBEDLET_AUTO_THREADS, results, &count);
    
    for (size_t i = 0; i < count; i++) {
        printf("Rank %zu: id=%zu, similarity=%.4f\n", 
               i + 1, results[i].id, results[i].score);
    }
    
    // Clean up
    embedlet_close(store, true);  // true = compact (remove trailing deletions)
    return 0;
}
```

### 3. Build

**Linux (GCC):**
```bash
gcc -O2 -mavx2 -mfma -o myapp myapp.c -lm -lpthread
```

**Windows (MSVC):**
```bat
cl /O2 /arch:AVX2 myapp.c
```

**Windows (MinGW64):**
```bash
gcc -O2 -mavx2 -mfma -o myapp.exe myapp.c -lm
```

See [Build Options](#build-options) for more details.

---

## Installation

embedlet is header-only. You have two options:

### Option A: Copy the Header

Just copy `include/embedlet.h` to your project's include directory.

### Option B: Use as a Subdirectory

```
your_project/
├── third_party/
│   └── embedlet/
│       └── include/
│           └── embedlet.h
└── src/
    └── main.c
```

Then compile with `-Ithird_party/embedlet/include`.

### Important: The Implementation

In **exactly one** `.c` file in your project, define `EMBEDLET_IMPLEMENTATION` before including the header:

```c
#define EMBEDLET_IMPLEMENTATION
#include "embedlet.h"
```

All other files should include the header without this define:

```c
#include "embedlet.h"  // Just declarations
```

---

## Build Options

### Linux / macOS (GCC or Clang)

```bash
# Release build with auto-detected SIMD
gcc -O3 -march=native -o myapp myapp.c -lm -lpthread

# Specific SIMD levels
gcc -O3 -mavx512f -mfma -o myapp myapp.c -lm -lpthread  # AVX-512
gcc -O3 -mavx2 -mfma -o myapp myapp.c -lm -lpthread     # AVX2 (recommended)
gcc -O3 -msse4.2 -o myapp myapp.c -lm -lpthread         # SSE4.2
gcc -O3 -o myapp myapp.c -lm -lpthread                  # Pure C fallback

# Debug build
gcc -O0 -g -o myapp myapp.c -lm -lpthread
```

### Windows (MSVC)

Run from a **Developer Command Prompt**:

```bat
@rem Release build with AVX2
cl /O2 /arch:AVX2 myapp.c

@rem Other SIMD options
cl /O2 /arch:AVX512 myapp.c   &:: AVX-512 (VS 2019 16.3+)
cl /O2 /arch:AVX myapp.c      &:: AVX
cl /O2 myapp.c                &:: SSE2 (default for x64)

@rem Debug build
cl /Od /Zi myapp.c
```

### Windows (MinGW64)

```bash
# Release with AVX2
gcc -O2 -mavx2 -mfma -o myapp.exe myapp.c -lm

# Other options same as Linux
```

### Using the Build Scripts

The repository includes build scripts for convenience:

```bash
# Linux
./scripts/build.sh gcc release native    # Auto-detect CPU
./scripts/build.sh gcc release avx2      # Force AVX2
./scripts/build.sh clang debug sse       # Debug with SSE only
```

```bat
@rem Windows
scripts\build.bat msvc release
scripts\build.bat mingw release
```

---

## API Reference

### Error Codes

All functions that can fail return an `int` status code:

| Code | Value | Meaning |
|------|-------|---------|
| `EMBEDLET_OK` | 0 | Success |
| `EMBEDLET_ERR_INVALID_ARG` | -1 | NULL pointer or invalid parameter |
| `EMBEDLET_ERR_INVALID_ID` | -2 | Embedding ID out of range |
| `EMBEDLET_ERR_FILE_OPEN` | -3 | Could not open or create file |
| `EMBEDLET_ERR_MMAP` | -4 | Memory mapping failed |
| `EMBEDLET_ERR_ALLOC` | -5 | Memory allocation failed |
| `EMBEDLET_ERR_TRUNCATE` | -6 | Could not resize file |
| `EMBEDLET_ERR_THREAD` | -7 | Thread pool creation failed |

### Thread Count Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `EMBEDLET_AUTO_THREADS` | 0 | Auto-detect CPU cores (max 8) |
| `EMBEDLET_SINGLE_THREAD` | 1 | Single-threaded operation |

---

### Types

#### `embedlet_store_t`

Opaque handle to an embedding store. Created by `embedlet_open()`, freed by `embedlet_close()`.

```c
embedlet_store_t *store;
```

#### `embedlet_result_t`

Result from a similarity search.

```c
typedef struct {
    size_t id;      // Index of the embedding
    float  score;   // Cosine similarity score (-1.0 to 1.0)
} embedlet_result_t;
```

---

### Functions

#### `embedlet_open`

```c
int embedlet_open(const char *path, size_t dims, embedlet_store_t **store_out);
```

Open an existing embedding store or create a new one.

**Parameters:**
- `path` — File path for the store (created if it doesn't exist)
- `dims` — Number of dimensions per embedding (must be > 0, must match existing file)
- `store_out` — Receives the store handle on success

**Returns:** `EMBEDLET_OK` on success, error code otherwise.

**Example:**
```c
embedlet_store_t *store;
int err = embedlet_open("vectors.db", 1536, &store);
if (err != EMBEDLET_OK) {
    fprintf(stderr, "Failed to open store: %d\n", err);
    return 1;
}
```

---

#### `embedlet_close`

```c
int embedlet_close(embedlet_store_t *store, bool compact);
```

Close a store and release all resources.

**Parameters:**
- `store` — Store handle
- `compact` — If `true`, truncate any trailing deleted (zeroed) embeddings

**Returns:** `EMBEDLET_OK` on success, error code otherwise.

**Example:**
```c
embedlet_close(store, true);  // Close and compact
```

---

#### `embedlet_count`

```c
size_t embedlet_count(const embedlet_store_t *store);
```

Get the number of embedding slots in the store. Note: this includes deleted (zeroed) slots that haven't been compacted.

**Parameters:**
- `store` — Store handle

**Returns:** Number of embeddings (file_size / (dims × sizeof(float))).

**Example:**
```c
printf("Store contains %zu embeddings\n", embedlet_count(store));
```

---

#### `embedlet_dims`

```c
size_t embedlet_dims(const embedlet_store_t *store);
```

Get the dimensionality of the store.

**Parameters:**
- `store` — Store handle

**Returns:** Number of dimensions per embedding.

---

#### `embedlet_append`

```c
int embedlet_append(embedlet_store_t *store, const float *data, bool reuse, size_t *id_out);
```

Add a new embedding to the store.

**Parameters:**
- `store` — Store handle
- `data` — Pointer to `dims` floats
- `reuse` — If `true`, reuse the first zeroed (deleted) slot; if `false`, always append at end
- `id_out` — Receives the assigned embedding ID

**Returns:** `EMBEDLET_OK` on success, error code otherwise.

**Notes:**
- When `reuse=false`, IDs are assigned sequentially (0, 1, 2, ...)
- When `reuse=true`, deleted slots may be reused, providing index stability
- The file grows automatically as needed

**Example:**
```c
float embedding[384] = { /* from your ML model */ };
size_t id;
int err = embedlet_append(store, embedding, false, &id);
if (err == EMBEDLET_OK) {
    printf("Stored at index %zu\n", id);
}
```

---

#### `embedlet_replace`

```c
int embedlet_replace(embedlet_store_t *store, size_t id, const float *data);
```

Replace the embedding at an existing index.

**Parameters:**
- `store` — Store handle
- `id` — Index to replace (must be < `embedlet_count()`)
- `data` — Pointer to `dims` floats

**Returns:** `EMBEDLET_OK` on success, `EMBEDLET_ERR_INVALID_ID` if out of range.

**Example:**
```c
float updated[384] = { /* new embedding */ };
embedlet_replace(store, 42, updated);
```

---

#### `embedlet_delete`

```c
int embedlet_delete(embedlet_store_t *store, size_t id);
```

Delete an embedding by zeroing all its values.

**Parameters:**
- `store` — Store handle
- `id` — Index to delete

**Returns:** `EMBEDLET_OK` on success, `EMBEDLET_ERR_INVALID_ID` if out of range.

**Notes:**
- Deleted embeddings are zeroed but remain in the file (preserving index stability)
- Trailing zeros are removed on `embedlet_close(store, true)` or `embedlet_compact()`
- Use `embedlet_append(store, data, true, &id)` to reuse deleted slots

**Example:**
```c
embedlet_delete(store, 42);
```

---

#### `embedlet_get`

```c
const float *embedlet_get(const embedlet_store_t *store, size_t id);
```

Get a read-only pointer to an embedding.

**Parameters:**
- `store` — Store handle
- `id` — Index to retrieve

**Returns:** Pointer to `dims` floats, or `NULL` if id is invalid.

**Warning:** The returned pointer is into the memory-mapped file. Do not modify it, and do not use it after the store is closed or compacted.

**Example:**
```c
const float *emb = embedlet_get(store, 42);
if (emb) {
    printf("First value: %f\n", emb[0]);
}
```

---

#### `embedlet_is_zeroed`

```c
bool embedlet_is_zeroed(const embedlet_store_t *store, size_t id);
```

Check if an embedding slot is zeroed (deleted).

**Parameters:**
- `store` — Store handle
- `id` — Index to check

**Returns:** `true` if all values are zero, `false` otherwise.

**Example:**
```c
if (embedlet_is_zeroed(store, 42)) {
    printf("Embedding 42 has been deleted\n");
}
```

---

#### `embedlet_similarity`

```c
float embedlet_similarity(const embedlet_store_t *store, const float *a, const float *b);
```

Compute cosine similarity between two embeddings.

**Parameters:**
- `store` — Store handle (used to get dimension count)
- `a` — First embedding
- `b` — Second embedding

**Returns:** Cosine similarity in the range [-1.0, 1.0]:
- `1.0` = identical direction
- `0.0` = orthogonal (unrelated)
- `-1.0` = opposite direction

**Example:**
```c
float sim = embedlet_similarity(store, emb_a, emb_b);
printf("Similarity: %.4f\n", sim);
```

---

#### `embedlet_similarity_raw`

```c
float embedlet_similarity_raw(const float *a, const float *b, size_t dims);
```

Compute cosine similarity without a store handle.

**Parameters:**
- `a` — First embedding
- `b` — Second embedding
- `dims` — Number of dimensions

**Returns:** Cosine similarity in the range [-1.0, 1.0].

**Example:**
```c
float sim = embedlet_similarity_raw(emb_a, emb_b, 384);
```

---

#### `embedlet_search`

```c
int embedlet_search(embedlet_store_t *store, const float *query, size_t n,
                    bool most_similar, int num_threads,
                    embedlet_result_t *results, size_t *count_out);
```

Find the top-N most or least similar embeddings to a query.

**Parameters:**
- `store` — Store handle
- `query` — Query embedding (`dims` floats)
- `n` — Maximum number of results to return
- `most_similar` — `true` for highest similarity, `false` for lowest
- `num_threads` — `EMBEDLET_AUTO_THREADS`, `EMBEDLET_SINGLE_THREAD`, or specific count
- `results` — Array of at least `n` `embedlet_result_t` to receive results
- `count_out` — Receives actual number of results (may be < n if store has fewer embeddings)

**Returns:** `EMBEDLET_OK` on success, error code otherwise.

**Notes:**
- Results are sorted by score (descending for most_similar, ascending for least_similar)
- Deleted (zeroed) embeddings are automatically skipped
- The thread pool is created lazily on first parallel search
- For small stores (< 1000 embeddings), single-threaded is often faster

**Example:**
```c
embedlet_result_t results[10];
size_t count;

int err = embedlet_search(store, query, 10, true, EMBEDLET_AUTO_THREADS, results, &count);
if (err == EMBEDLET_OK) {
    for (size_t i = 0; i < count; i++) {
        printf("%zu. id=%zu score=%.4f\n", i+1, results[i].id, results[i].score);
    }
}
```

---

#### `embedlet_compact`

```c
int embedlet_compact(embedlet_store_t *store);
```

Remove trailing deleted embeddings from the file.

**Parameters:**
- `store` — Store handle

**Returns:** `EMBEDLET_OK` on success, error code otherwise.

**Notes:**
- Only trailing zeros are removed; zeros in the middle are preserved
- This is automatically called by `embedlet_close(store, true)`
- After compaction, `embedlet_count()` will return a smaller value

**Example:**
```c
printf("Before: %zu embeddings\n", embedlet_count(store));
embedlet_compact(store);
printf("After: %zu embeddings\n", embedlet_count(store));
```

---

## File Format

The store file is simply a flat array of `float` values:

```
[embedding_0: dims × float32] [embedding_1: dims × float32] ... [embedding_n: dims × float32]
```

- No header, no metadata
- File size = count × dims × 4 bytes
- Embeddings are stored in the order they were appended
- Deleted embeddings are all zeros

This means you can:
- Memory-map the file directly from other languages
- Inspect it with hex editors
- Copy/concat files (if same dimensionality)

---

## Performance Tips

### SIMD Selection

For best performance, compile with the highest SIMD level your deployment CPUs support:

| CPU Generation | Recommended Flag | Speedup vs. Pure C |
|----------------|------------------|---------------------|
| Zen 4, Sapphire Rapids | `-mavx512f -mfma` or `/arch:AVX512` | ~8-16× |
| Haswell+, Zen 1-3 | `-mavx2 -mfma` or `/arch:AVX2` | ~4-8× |
| Any x86-64 | `-msse4.2` or (default) | ~2-4× |
| Unknown/mixed | (no flags) | 1× (baseline) |

Use `-march=native` for local builds to auto-detect.

### Threading

- **Small stores (< 10K embeddings):** Use `EMBEDLET_SINGLE_THREAD`
- **Large stores (> 100K embeddings):** Use `EMBEDLET_AUTO_THREADS`
- **Latency-sensitive:** Benchmark both; thread overhead can hurt small queries

### Dimensionality

- Higher dimensions = more accurate similarity but slower search
- Common sizes: 384, 768, 1024, 1536, 3072
- embedlet handles any dimension efficiently due to SIMD

### Memory Usage

The entire store is memory-mapped, so:
- Active portions are paged into RAM on demand
- You can have stores larger than physical RAM
- First access to "cold" pages may be slow (disk I/O)

---

## Thread Safety

- **Reads are lock-free:** Multiple threads can call `embedlet_get()`, `embedlet_similarity()`, and `embedlet_search()` concurrently
- **Writes are serialized:** `embedlet_append()`, `embedlet_replace()`, `embedlet_delete()`, and `embedlet_compact()` acquire an internal mutex
- **Safe pattern:** One writer thread + multiple reader threads

---

## Limitations

- **No indexing:** Search is brute-force O(n). For millions of embeddings, consider adding an ANN index (HNSW, IVF, etc.)
- **Fixed dimensions:** All embeddings in a store must have the same dimensionality
- **No metadata:** embedlet only stores vectors. Keep a separate mapping of IDs to your application data
- **32-bit floats only:** No fp16 or quantized vectors (yet)

---

## Example: Semantic Search

Here's a more complete example showing how you might build a simple semantic search system:

```c
#define EMBEDLET_IMPLEMENTATION
#include "embedlet.h"
#include <stdio.h>
#include <stdlib.h>

// Pretend this calls an embedding API
void get_embedding(const char *text, float *out, size_t dims) {
    // In reality: call OpenAI, Cohere, or run a local model
    // For demo, just fill with dummy values
    for (size_t i = 0; i < dims; i++) {
        out[i] = (float)(text[i % 64]) / 255.0f;
    }
}

int main(void) {
    const size_t DIMS = 384;
    embedlet_store_t *store;
    
    // Open or create the store
    if (embedlet_open("documents.db", DIMS, &store) != EMBEDLET_OK) {
        fprintf(stderr, "Failed to open store\n");
        return 1;
    }
    
    // Sample documents (in practice, load from a file or database)
    const char *documents[] = {
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Neural networks are inspired by biological brains",
        "The weather today is sunny and warm",
    };
    size_t num_docs = sizeof(documents) / sizeof(documents[0]);
    
    // Index documents (only if store is empty)
    if (embedlet_count(store) == 0) {
        printf("Indexing %zu documents...\n", num_docs);
        float *emb = malloc(DIMS * sizeof(float));
        
        for (size_t i = 0; i < num_docs; i++) {
            get_embedding(documents[i], emb, DIMS);
            size_t id;
            embedlet_append(store, emb, false, &id);
            printf("  [%zu] %s\n", id, documents[i]);
        }
        
        free(emb);
    }
    
    // Search
    const char *query = "What is AI?";
    printf("\nSearching for: \"%s\"\n\n", query);
    
    float *query_emb = malloc(DIMS * sizeof(float));
    get_embedding(query, query_emb, DIMS);
    
    embedlet_result_t results[3];
    size_t count;
    
    embedlet_search(store, query_emb, 3, true, EMBEDLET_AUTO_THREADS, results, &count);
    
    printf("Top %zu results:\n", count);
    for (size_t i = 0; i < count; i++) {
        printf("  %zu. [%.4f] %s\n", 
               i + 1, 
               results[i].score, 
               documents[results[i].id]);
    }
    
    free(query_emb);
    embedlet_close(store, false);
    return 0;
}
```

---

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

For more information, see [unlicense.org](http://unlicense.org/) or the `UNLICENSE` file.

---

## Author

coeurnix

---

## Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest features
- Submit pull requests

Since this is public domain, your contributions will also be released under the Unlicense.
