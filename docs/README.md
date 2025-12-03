# API Reference for embedlet

This document provides detailed API documentation for the embedlet C library.

## Error Codes

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

## Thread Count Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `EMBEDLET_AUTO_THREADS` | 0 | Auto-detect CPU cores (max 8) |
| `EMBEDLET_SINGLE_THREAD` | 1 | Single-threaded operation |

---

## Types

### `embedlet_store_t`

Opaque handle to an embedding store. Created by `embedlet_open()`, freed by `embedlet_close()`.

```c
embedlet_store_t *store;
```

### `embedlet_result_t`

Result from a similarity search.

```c
typedef struct {
    size_t id;      // Index of the embedding
    float  score;   // Cosine similarity score (-1.0 to 1.0)
} embedlet_result_t;
```

---

## Functions

### `embedlet_open`

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

### `embedlet_close`

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

### `embedlet_count`

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

### `embedlet_dims`

```c
size_t embedlet_dims(const embedlet_store_t *store);
```

Get the dimensionality of the store.

**Parameters:**
- `store` — Store handle

**Returns:** Number of dimensions per embedding.

---

### `embedlet_append`

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

### `embedlet_replace`

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

### `embedlet_delete`

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

### `embedlet_get`

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

### `embedlet_is_zeroed`

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

### `embedlet_similarity`

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

### `embedlet_similarity_raw`

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

### `embedlet_search`

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

### `embedlet_compact`

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
