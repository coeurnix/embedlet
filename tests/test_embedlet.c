// test_embedlet.c
/**
 * @file test_embedlet.c
 * @brief Comprehensive unit tests for embedlet library.
 *
 * Assumes existence of embedding-000.dat through embedding-149.dat,
 * each containing one 1024-dimensional float32 embedding.
 *
 * Compile:
 *   Linux:   gcc -O2 -mavx2 -mfma test_embedlet.c -o test -lm -lpthread
 *   Windows: cl /O2 /arch:AVX2 test_embedlet.c /link
 */

#define EMBEDLET_IMPLEMENTATION
#include "../include/embedlet.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

#define TEST_DIMS 1024
#define TEST_NUM_FILES 150
#define TEST_STORE_PATH "test_store.emb"

/* Load a single embedding from file */
static int load_embedding(const char *path, float *out, size_t dims) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  size_t read = fread(out, sizeof(float), dims, f);
  fclose(f);
  return (read == dims) ? 0 : -1;
}

/* Generate embedding filename */
static void get_embedding_path(int idx, char *buf, size_t bufsize) {
  snprintf(buf, bufsize, "../sample_data/embedding-%03d.dat", idx);
}

/* Test: Open and create store */
static void test_open_create(void) {
  printf("Testing open/create...\n");

  /* Remove existing test file */
  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  int err = embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);
  (void)err; /* Used for assertion */
  assert(err == EMBEDLET_OK);
  assert(store != NULL);
  assert(embedlet_count(store) == 0);
  assert(embedlet_dims(store) == TEST_DIMS);

  err = embedlet_close(store, false);
  assert(err == EMBEDLET_OK);

  /* Reopen existing */
  err = embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);
  assert(err == EMBEDLET_OK);
  assert(embedlet_count(store) == 0);

  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Append embeddings */
static void test_append(void) {
  printf("Testing append...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  int err = embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);
  (void)err; /* Used for assertion */
  assert(err == EMBEDLET_OK);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];
  for (int i = 0; i < 10; i++) {
    get_embedding_path(i, path, sizeof(path));
    err = load_embedding(path, emb, TEST_DIMS);
    assert(err == 0);

    size_t id;
    err = embedlet_append(store, emb, false, &id);
    assert(err == EMBEDLET_OK);
    assert(id == (size_t)i);
  }

  assert(embedlet_count(store) == 10);

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Append with reuse */
static void test_append_reuse(void) {
  printf("Testing append with reuse...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  int err = embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);
  (void)err; /* Used for assertion */
  assert(err == EMBEDLET_OK);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Add 5 embeddings */
  for (int i = 0; i < 5; i++) {
    get_embedding_path(i, path, sizeof(path));
    load_embedding(path, emb, TEST_DIMS);
    size_t id;
    embedlet_append(store, emb, false, &id);
    assert(id == (size_t)i);
  }

  /* Delete embedding 2 */
  err = embedlet_delete(store, 2);
  assert(err == EMBEDLET_OK);
  assert(embedlet_is_zeroed(store, 2));

  /* Append with reuse=true should reuse slot 2 */
  get_embedding_path(10, path, sizeof(path));
  load_embedding(path, emb, TEST_DIMS);
  size_t id;
  err = embedlet_append(store, emb, true, &id);
  assert(err == EMBEDLET_OK);
  assert(id == 2);

  /* Append with reuse=false should append at end */
  get_embedding_path(11, path, sizeof(path));
  load_embedding(path, emb, TEST_DIMS);
  err = embedlet_append(store, emb, false, &id);
  assert(err == EMBEDLET_OK);
  assert(id == 5);

  assert(embedlet_count(store) == 6);

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Replace embedding */
static void test_replace(void) {
  printf("Testing replace...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb1 = (float *)malloc(TEST_DIMS * sizeof(float));
  float *emb2 = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb1 && emb2);

  char path[64];
  get_embedding_path(0, path, sizeof(path));
  load_embedding(path, emb1, TEST_DIMS);

  size_t id;
  embedlet_append(store, emb1, false, &id);
  assert(id == 0);

  /* Replace with different embedding */
  get_embedding_path(1, path, sizeof(path));
  load_embedding(path, emb2, TEST_DIMS);

  int err = embedlet_replace(store, 0, emb2);
  assert(err == EMBEDLET_OK);

  /* Verify replacement */
  const float *stored = embedlet_get(store, 0);
  assert(stored != NULL);
  for (size_t i = 0; i < TEST_DIMS; i++) {
    assert(stored[i] == emb2[i]);
  }

  /* Test invalid ID */
  err = embedlet_replace(store, 999, emb1);
  assert(err == EMBEDLET_ERR_INVALID_ID);

  free(emb1);
  free(emb2);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Delete and compact */
static void test_delete_compact(void) {
  printf("Testing delete and compact...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Add 10 embeddings */
  for (int i = 0; i < 10; i++) {
    get_embedding_path(i, path, sizeof(path));
    load_embedding(path, emb, TEST_DIMS);
    size_t id;
    embedlet_append(store, emb, false, &id);
  }
  assert(embedlet_count(store) == 10);

  /* Delete last 3 (indices 7, 8, 9) */
  embedlet_delete(store, 7);
  embedlet_delete(store, 8);
  embedlet_delete(store, 9);

  /* Delete one in middle (index 3) */
  embedlet_delete(store, 3);

  assert(embedlet_is_zeroed(store, 3));
  assert(embedlet_is_zeroed(store, 7));
  assert(!embedlet_is_zeroed(store, 5));

  /* Compact should truncate trailing zeros */
  int err = embedlet_compact(store);
  assert(err == EMBEDLET_OK);
  assert(embedlet_count(store) == 7);

  /* Middle zero should remain */
  assert(embedlet_is_zeroed(store, 3));

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Pairwise similarity */
static void test_similarity(void) {
  printf("Testing similarity...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb0 = (float *)malloc(TEST_DIMS * sizeof(float));
  float *emb1 = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb0 && emb1);

  char path[64];
  get_embedding_path(0, path, sizeof(path));
  load_embedding(path, emb0, TEST_DIMS);
  get_embedding_path(1, path, sizeof(path));
  load_embedding(path, emb1, TEST_DIMS);

  /* Self-similarity should be ~1.0 */
  float self_sim = embedlet_similarity(store, emb0, emb0);
  assert(self_sim > 0.999f && self_sim <= 1.001f);

  /* Cross-similarity should be in valid range */
  float cross_sim = embedlet_similarity(store, emb0, emb1);
  assert(cross_sim >= -1.0f && cross_sim <= 1.0f);

  /* Test raw function */
  float raw_sim = embedlet_similarity_raw(emb0, emb1, TEST_DIMS);
  assert(fabsf(raw_sim - cross_sim) < 0.0001f);

  free(emb0);
  free(emb1);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Top-N search (single-threaded) */
static void test_search_single(void) {
  printf("Testing top-N search (single-threaded)...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Load all 150 embeddings */
  for (int i = 0; i < TEST_NUM_FILES; i++) {
    get_embedding_path(i, path, sizeof(path));
    int err = load_embedding(path, emb, TEST_DIMS);
    assert(err == 0);
    size_t id;
    err = embedlet_append(store, emb, false, &id);
    assert(err == EMBEDLET_OK);
  }

  assert(embedlet_count(store) == TEST_NUM_FILES);

  /* Query with embedding 0, find top 5 most similar */
  get_embedding_path(0, path, sizeof(path));
  load_embedding(path, emb, TEST_DIMS);

  embedlet_result_t results[5];
  size_t count;
  int err = embedlet_search(store, emb, 5, true, EMBEDLET_SINGLE_THREAD,
                            results, &count);
  assert(err == EMBEDLET_OK);
  assert(count == 5);

  /* First result should be embedding 0 itself (similarity ~1.0) */
  assert(results[0].id == 0);
  assert(results[0].score > 0.99f);

  /* Results should be sorted descending */
  for (size_t i = 1; i < count; i++) {
    assert(results[i].score <= results[i - 1].score);
  }

  /* Test least similar */
  err = embedlet_search(store, emb, 5, false, EMBEDLET_SINGLE_THREAD, results,
                        &count);
  assert(err == EMBEDLET_OK);
  assert(count == 5);

  /* Results should be sorted ascending */
  for (size_t i = 1; i < count; i++) {
    assert(results[i].score >= results[i - 1].score);
  }

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Top-N search (multi-threaded) */
static void test_search_multi(void) {
  printf("Testing top-N search (multi-threaded)...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Load all 150 embeddings */
  for (int i = 0; i < TEST_NUM_FILES; i++) {
    get_embedding_path(i, path, sizeof(path));
    load_embedding(path, emb, TEST_DIMS);
    size_t id;
    embedlet_append(store, emb, false, &id);
  }

  /* Query with embedding 50 */
  get_embedding_path(50, path, sizeof(path));
  load_embedding(path, emb, TEST_DIMS);

  /* Run with auto threads */
  embedlet_result_t results_auto[10];
  size_t count_auto;
  int err = embedlet_search(store, emb, 10, true, EMBEDLET_AUTO_THREADS,
                            results_auto, &count_auto);
  assert(err == EMBEDLET_OK);
  assert(count_auto == 10);

  /* Run single-threaded for comparison */
  embedlet_result_t results_single[10];
  size_t count_single;
  err = embedlet_search(store, emb, 10, true, EMBEDLET_SINGLE_THREAD,
                        results_single, &count_single);
  assert(err == EMBEDLET_OK);
  assert(count_single == 10);

  /* Results should match */
  for (size_t i = 0; i < count_auto; i++) {
    assert(results_auto[i].id == results_single[i].id);
    assert(fabsf(results_auto[i].score - results_single[i].score) < 0.0001f);
  }

  /* First result should be embedding 50 */
  assert(results_auto[0].id == 50);

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Thread safety (concurrent reads) */
static void test_thread_safety(void) {
  printf("Testing thread safety...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Load 50 embeddings */
  for (int i = 0; i < 50; i++) {
    get_embedding_path(i, path, sizeof(path));
    load_embedding(path, emb, TEST_DIMS);
    size_t id;
    embedlet_append(store, emb, false, &id);
  }

  /* Perform multiple concurrent searches */
  for (int round = 0; round < 5; round++) {
    get_embedding_path(round * 10, path, sizeof(path));
    load_embedding(path, emb, TEST_DIMS);

    embedlet_result_t results[5];
    size_t count;
    int err = embedlet_search(store, emb, 5, true, 4, results, &count);
    assert(err == EMBEDLET_OK);
    assert(count == 5);
    assert(results[0].id == (size_t)(round * 10));
  }

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Persistence */
static void test_persistence(void) {
  printf("Testing persistence...\n");

  remove(TEST_STORE_PATH);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Create and populate store */
  {
    embedlet_store_t *store = NULL;
    embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

    for (int i = 0; i < 20; i++) {
      get_embedding_path(i, path, sizeof(path));
      load_embedding(path, emb, TEST_DIMS);
      size_t id;
      embedlet_append(store, emb, false, &id);
    }

    embedlet_close(store, false);
  }

  /* Reopen and verify */
  {
    embedlet_store_t *store = NULL;
    embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

    assert(embedlet_count(store) == 20);

    /* Verify first embedding */
    get_embedding_path(0, path, sizeof(path));
    load_embedding(path, emb, TEST_DIMS);
    const float *stored = embedlet_get(store, 0);
    assert(stored != NULL);

    float sim = embedlet_similarity(store, emb, stored);
    assert(sim > 0.999f);

    embedlet_close(store, false);
  }

  free(emb);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Close with compact */
static void test_close_compact(void) {
  printf("Testing close with compact...\n");

  remove(TEST_STORE_PATH);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];

  /* Create store with trailing zeros */
  {
    embedlet_store_t *store = NULL;
    embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

    for (int i = 0; i < 10; i++) {
      get_embedding_path(i, path, sizeof(path));
      load_embedding(path, emb, TEST_DIMS);
      size_t id;
      embedlet_append(store, emb, false, &id);
    }

    /* Delete last 3 */
    embedlet_delete(store, 7);
    embedlet_delete(store, 8);
    embedlet_delete(store, 9);

    /* Close with compact=true */
    embedlet_close(store, true);
  }

  /* Reopen and verify truncation */
  {
    embedlet_store_t *store = NULL;
    embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

    assert(embedlet_count(store) == 7);

    embedlet_close(store, false);
  }

  free(emb);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Edge cases */
static void test_edge_cases(void) {
  printf("Testing edge cases...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  int err;

  /* Invalid arguments */
  err = embedlet_open(NULL, TEST_DIMS, &store);
  assert(err == EMBEDLET_ERR_INVALID_ARG);

  err = embedlet_open(TEST_STORE_PATH, 0, &store);
  assert(err == EMBEDLET_ERR_INVALID_ARG);

  err = embedlet_open(TEST_STORE_PATH, TEST_DIMS, NULL);
  assert(err == EMBEDLET_ERR_INVALID_ARG);

  /* Valid open */
  err = embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);
  assert(err == EMBEDLET_OK);

  /* Search on empty store */
  float query[TEST_DIMS] = {0};
  embedlet_result_t results[5];
  size_t count;
  err = embedlet_search(store, query, 5, true, 1, results, &count);
  assert(err == EMBEDLET_OK);
  assert(count == 0);

  /* Get on empty store */
  const float *ptr = embedlet_get(store, 0);
  assert(ptr == NULL);

  /* Delete invalid ID */
  err = embedlet_delete(store, 0);
  assert(err == EMBEDLET_ERR_INVALID_ID);

  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

/* Test: Large batch append */
static void test_batch_append(void) {
  printf("Testing batch append (all 150 files)...\n");

  remove(TEST_STORE_PATH);

  embedlet_store_t *store = NULL;
  embedlet_open(TEST_STORE_PATH, TEST_DIMS, &store);

  float *emb = (float *)malloc(TEST_DIMS * sizeof(float));
  assert(emb != NULL);

  char path[64];
  clock_t start = clock();

  for (int i = 0; i < TEST_NUM_FILES; i++) {
    get_embedding_path(i, path, sizeof(path));
    int err = load_embedding(path, emb, TEST_DIMS);
    assert(err == 0);

    size_t id;
    err = embedlet_append(store, emb, false, &id);
    assert(err == EMBEDLET_OK);
    assert(id == (size_t)i);
  }

  clock_t end = clock();
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

  assert(embedlet_count(store) == TEST_NUM_FILES);
  printf("  Appended %d embeddings in %.3f seconds\n", TEST_NUM_FILES, elapsed);

  free(emb);
  embedlet_close(store, false);
  remove(TEST_STORE_PATH);

  printf("  PASSED\n");
}

int main(void) {
  printf("=== Embedlet Unit Tests ===\n\n");

  test_open_create();
  test_append();
  test_append_reuse();
  test_replace();
  test_delete_compact();
  test_similarity();
  test_search_single();
  test_search_multi();
  test_thread_safety();
  test_persistence();
  test_close_compact();
  test_edge_cases();
  test_batch_append();

  printf("\n=== All tests PASSED ===\n");
  return 0;
}