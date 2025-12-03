// example.c
/**
 * @file example.c
 * @brief Minimal usage example for embedlet library.
 *
 * Demonstrates: open, append, search, delete, replace, compact.
 * Assumes embedding-000.dat through embedding-149.dat exist.
 *
 * Compile:
 *   Linux:   gcc -O2 -mavx2 -mfma example.c -o example -lm -lpthread
 *   Windows: cl /O2 /arch:AVX2 example.c /link
 */

#define EMBEDLET_IMPLEMENTATION
#include "../include/embedlet.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DIMS 1024
#define STORE_PATH "example_store.emb"

/* Load embedding from file */
static int load_embedding(const char *path, float *out, size_t dims) {
  FILE *f;
  errno_t err = fopen_s(&f, path, "rb");
  if (err != 0) {
    fprintf(stderr, "Failed to open: %s\n", path);
    return -1;
  }
  size_t read = fread(out, sizeof(float), dims, f);
  fclose(f);
  return (read == dims) ? 0 : -1;
}

int main(void) {
  embedlet_store_t *store = NULL;
  int err;

  printf("=== Embedlet Example ===\n\n");

  /* Remove any existing store */
  remove(STORE_PATH);

  /* Open/create store */
  printf("Opening store with %d dimensions...\n", DIMS);
  err = embedlet_open(STORE_PATH, DIMS, &store);
  if (err != EMBEDLET_OK) {
    fprintf(stderr, "Failed to open store: %d\n", err);
    return 1;
  }
  printf("Store opened. Initial count: %zu\n\n", embedlet_count(store));

  /* Allocate buffer for embeddings */
  float *emb = (float *)malloc(DIMS * sizeof(float));
  if (!emb) {
    fprintf(stderr, "Allocation failed\n");
    embedlet_close(store, false);
    return 1;
  }

  /* Append first 20 embeddings */
  printf("Appending embeddings 0-19...\n");
  for (int i = 0; i < 20; i++) {
    char path[64];
    snprintf(path, sizeof(path), "../sample_data/embedding-%03d.dat", i);

    if (load_embedding(path, emb, DIMS) != 0) {
      fprintf(stderr, "Warning: Could not load %s\n", path);
      continue;
    }

    size_t id;
    err = embedlet_append(store, emb, false, &id);
    if (err == EMBEDLET_OK) {
      printf("  Appended embedding %d -> id %zu\n", i, id);
    }
  }
  printf("Count after append: %zu\n\n", embedlet_count(store));

  /* Perform top-5 similarity search */
  printf("Loading query from ../sample_data/embedding-000.dat...\n");
  if (load_embedding("../sample_data/embedding-000.dat", emb, DIMS) != 0) {
    fprintf(stderr, "Failed to load query embedding\n");
    free(emb);
    embedlet_close(store, false);
    return 1;
  }

  printf("\n--- Top-5 Most Similar (single-threaded) ---\n");
  embedlet_result_t results[5];
  size_t result_count;

  err = embedlet_search(store, emb, 5, true, EMBEDLET_SINGLE_THREAD, results,
                        &result_count);
  if (err == EMBEDLET_OK) {
    for (size_t i = 0; i < result_count; i++) {
      printf("  Rank %zu: id=%zu, score=%.6f\n", i + 1, results[i].id,
             results[i].score);
    }
  }

  printf("\n--- Top-5 Most Similar (multi-threaded, auto) ---\n");
  err = embedlet_search(store, emb, 5, true, EMBEDLET_AUTO_THREADS, results,
                        &result_count);
  if (err == EMBEDLET_OK) {
    for (size_t i = 0; i < result_count; i++) {
      printf("  Rank %zu: id=%zu, score=%.6f\n", i + 1, results[i].id,
             results[i].score);
    }
  }

  printf("\n--- Top-5 Least Similar ---\n");
  err = embedlet_search(store, emb, 5, false, EMBEDLET_SINGLE_THREAD, results,
                        &result_count);
  if (err == EMBEDLET_OK) {
    for (size_t i = 0; i < result_count; i++) {
      printf("  Rank %zu: id=%zu, score=%.6f\n", i + 1, results[i].id,
             results[i].score);
    }
  }

  /* Delete some embeddings */
  printf("\nDeleting embeddings at indices 5, 10, 15...\n");
  embedlet_delete(store, 5);
  embedlet_delete(store, 10);
  embedlet_delete(store, 15);

  printf("Is index 5 zeroed? %s\n",
         embedlet_is_zeroed(store, 5) ? "yes" : "no");
  printf("Is index 6 zeroed? %s\n",
         embedlet_is_zeroed(store, 6) ? "yes" : "no");

  /* Replace an embedding */
  printf("\nReplacing embedding at index 3 with embedding-100.dat...\n");
  if (load_embedding("../sample_data/embedding-100.dat", emb, DIMS) == 0) {
    err = embedlet_replace(store, 3, emb);
    if (err == EMBEDLET_OK) {
      printf("  Replace successful\n");
    }
  }

  /* Append with reuse (should fill in a deleted slot) */
  printf("\nAppending with reuse=true...\n");
  if (load_embedding("../sample_data/embedding-050.dat", emb, DIMS) == 0) {
    size_t id;
    err = embedlet_append(store, emb, true, &id);
    if (err == EMBEDLET_OK) {
      printf("  Appended to id %zu (reused deleted slot)\n", id);
    }
  }

  /* Delete trailing embeddings for compaction demo */
  printf("\nDeleting trailing embeddings (18, 19) for compaction...\n");
  embedlet_delete(store, 18);
  embedlet_delete(store, 19);
  printf("Count before compact: %zu\n", embedlet_count(store));

  /* Compact */
  printf("\nCompacting store...\n");
  err = embedlet_compact(store);
  if (err == EMBEDLET_OK) {
    printf("Count after compact: %zu\n", embedlet_count(store));
  }

  /* Final search */
  printf("\n--- Final Top-5 Search ---\n");
  if (load_embedding("../sample_data/embedding-000.dat", emb, DIMS) == 0) {
    err = embedlet_search(store, emb, 5, true, EMBEDLET_AUTO_THREADS, results,
                          &result_count);
    if (err == EMBEDLET_OK) {
      for (size_t i = 0; i < result_count; i++) {
        printf("  Rank %zu: id=%zu, score=%.6f\n", i + 1, results[i].id,
               results[i].score);
      }
    }
  }

  /* Pairwise similarity */
  printf("\n--- Pairwise Similarity ---\n");
  float *emb_a = (float *)malloc(DIMS * sizeof(float));
  float *emb_b = (float *)malloc(DIMS * sizeof(float));
  if (emb_a && emb_b) {
    load_embedding("../sample_data/embedding-000.dat", emb_a, DIMS);
    load_embedding("../sample_data/embedding-001.dat", emb_b, DIMS);

    float sim = embedlet_similarity(store, emb_a, emb_b);
    printf("  Similarity(emb-000, emb-001) = %.6f\n", sim);

    sim = embedlet_similarity(store, emb_a, emb_a);
    printf("  Similarity(emb-000, emb-000) = %.6f (self)\n", sim);

    free(emb_a);
    free(emb_b);
  }

  /* Close with compaction */
  printf("\nClosing store with final compaction...\n");
  err = embedlet_close(store, true);
  if (err == EMBEDLET_OK) {
    printf("Store closed successfully.\n");
  }

  free(emb);

  /* Cleanup test file */
  remove(STORE_PATH);

  /* === Benchmark Phase === */
  printf("\n=== Embedlet Benchmark Phase ===\n");
#if defined(__i386__) || defined(_M_IX86)
  printf("Creating store with embeddings (32-bit build)\n");
#else
  printf("Creating store with 500,000 embeddings (~2GB)\n");
#endif

  /* Remove any existing benchmark store */
  remove(STORE_PATH);

  /* Open new store */
  embedlet_store_t *bench_store = NULL;
  err = embedlet_open(STORE_PATH, DIMS, &bench_store);
  if (err != EMBEDLET_OK) {
    fprintf(stderr, "Failed to open benchmark store: %d\n", err);
    return 1;
  }
  printf("Benchmark store opened. Initial count: %zu\n",
         embedlet_count(bench_store));

  /* Allocate new buffer for benchmark queries (previous emb was freed) */
  emb = (float *)malloc(DIMS * sizeof(float));
  if (!emb) {
    fprintf(stderr, "Allocation failed\n");
    embedlet_close(bench_store, false);
    return 1;
  }

  /* Allocate buffer for template embeddings */
  const int num_templates = 150;
  float *templates = (float *)malloc(num_templates * DIMS * sizeof(float));
  if (!templates) {
    fprintf(stderr, "Allocation failed for templates\n");
    embedlet_close(bench_store, false);
    return 1;
  }

  /* Load template embeddings */
  printf("Loading template embeddings...\n");
  for (int i = 0; i < num_templates; i++) {
    char path[64];
    snprintf(path, sizeof(path), "../sample_data/embedding-%03d.dat", i);
    if (load_embedding(path, templates + i * DIMS, DIMS) != 0) {
      fprintf(stderr, "Warning: Could not load %s\n", path);
      memset(templates + i * DIMS, 0, DIMS * sizeof(float)); // zero padding
    }
  }

  /* Fill store with embeddings by cycling through templates */
#if defined(__i386__) || defined(_M_IX86)
  /* 32-bit build - limit to avoid exceeding 2GB address space */
  const int total_embeddings = 200000;
  printf("Note: 32-bit build detected, using %d embeddings to stay under 2GB "
         "limit\n",
         total_embeddings);
  printf("      For full 500k benchmark, use 64-bit compiler (MinGW-w64 "
         "x86_64)\n");
#else
  /* 64-bit build - use full 500k */
  const int total_embeddings = 500000;
#endif
  printf("Filling store with %d embeddings...\n", total_embeddings);

  clock_t fill_start = clock();
  for (int i = 0; i < total_embeddings; i++) {
    size_t id;
    err = embedlet_append(bench_store, templates + (i % num_templates) * DIMS,
                          false, &id);
    if (err != EMBEDLET_OK) {
      fprintf(stderr, "Append failed at %d: %d\n", i, err);
      free(templates);
      embedlet_close(bench_store, false);
      remove(STORE_PATH);
      return 1;
    }
  }
  double fill_time = (double)(clock() - fill_start) / CLOCKS_PER_SEC;
  printf("Time to fill %d embeddings: %.2f seconds\n", total_embeddings,
         fill_time);

  /* Load query embedding */
  fprintf(stderr, "Load query from ../sample_data/embedding-000.dat...\n");
  if (load_embedding("../sample_data/embedding-000.dat", emb, DIMS) != 0) {
    fprintf(stderr, "Failed to load query embedding\n");
    free(templates);
    embedlet_close(bench_store, false);
    remove(STORE_PATH);
    return 1;
  }

  /* Time single-threaded top-5 search */
  printf("\nTiming single-threaded top-5 search (20x)...\n");
  clock_t search_start = clock();
  for (int i = 0; i < 20; i++) {
    err = embedlet_search(bench_store, emb, 5, true, EMBEDLET_SINGLE_THREAD,
                          results, &result_count);
  }
  double search_single_time = (double)(clock() - search_start) / CLOCKS_PER_SEC;
  if (err != EMBEDLET_OK) {
    fprintf(stderr, "Search failed: %d\n", err);
  } else {
    printf("Single-threaded search time: %.2f seconds\n", search_single_time);
  }

  /* Time multi-threaded top-5 search */
  printf("\nTiming multi-threaded top-5 search (20x)...\n");
  search_start = clock();
  for (int i = 0; i < 20; i++) {
    err = embedlet_search(bench_store, emb, 5, true, 2, results, &result_count);
  }
  double search_multi_time = (double)(clock() - search_start) / CLOCKS_PER_SEC;
  if (err != EMBEDLET_OK) {
    fprintf(
        stderr,
        "Multi-threaded search failed: %d (falling back to single-threaded)\n",
        err);
    /* Fallback to single-threaded */
    search_start = clock();
    err = embedlet_search(bench_store, emb, 5, true, EMBEDLET_SINGLE_THREAD,
                          results, &result_count);
    search_multi_time = (double)(clock() - search_start) / CLOCKS_PER_SEC;
    if (err == EMBEDLET_OK) {
      printf("Fallback single-threaded search time: %.2f seconds\n",
             search_multi_time);
    } else {
      printf("Fallback search also failed: %d\n", err);
      search_multi_time = -1.0;
    }
  } else {
    printf("Multi-threaded search time: %.2f seconds\n", search_multi_time);
  }

  /* Time replace in the middle */
  size_t middle_id = total_embeddings / 2;
  printf("\nTiming replace at middle index %zu...\n", middle_id);

  /* Load replacement embedding */
  float *replace_emb = emb; // reuse emb buffer
  if (load_embedding("../sample_data/embedding-100.dat", replace_emb, DIMS) !=
      0) {
    fprintf(stderr, "Failed to load replacement embedding\n");
    free(templates);
    embedlet_close(bench_store, false);
    remove(STORE_PATH);
    return 1;
  }

  clock_t replace_start = clock();
  err = embedlet_replace(bench_store, middle_id, replace_emb);
  double replace_time = (double)(clock() - replace_start) / CLOCKS_PER_SEC;
  if (err != EMBEDLET_OK) {
    fprintf(stderr, "Replace failed: %d\n", err);
  } else {
    printf("Replace time: %.2f seconds\n", replace_time);
  }

  /* Time compacting close */
  printf("\nTiming compacting close...\n");
  clock_t close_start = clock();
  err = embedlet_close(bench_store, true);
  double close_time = (double)(clock() - close_start) / CLOCKS_PER_SEC;
  if (err != EMBEDLET_OK) {
    fprintf(stderr, "Close failed: %d\n", err);
  } else {
    printf("Compacting close time: %.2f seconds\n", close_time);
  }

  free(templates);

  /* Free benchmark buffer */
  free(emb);

  /* Cleanup benchmark file */
  remove(STORE_PATH);

  printf("\n=== Benchmark Complete ===\n");

  printf("\n=== Example Complete ===\n");
  return 0;
}
