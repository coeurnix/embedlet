// embedlet.h
/**
 * @file embedlet.h
 * @brief Single-file, header-only C11 library for memory-mapped float32
 * embeddings.
 *
 * Provides storage, retrieval, and similarity search for fixed-dimensional
 * float32 embeddings using memory-mapped files. Thread-safe with optional
 * multithreaded queries. Uses a portable C implementation with optional
 * SSE2 acceleration when available.
 *
 * Basic Usage:
 *   #define EMBEDLET_IMPLEMENTATION
 *   #include "embedlet.h"
 *
 * Build Instructions:
 *
 *   MSVC:
 *     cl /O2 /DEMBEDLET_IMPLEMENTATION your_program.c
 *
 *   GCC/Clang:
 *     gcc -O3 -DEMBEDLET_IMPLEMENTATION your_program.c -o your_program -lm
 */

#ifndef EMBEDLET_H
#define EMBEDLET_H

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Platform Detection
 *============================================================================*/

#if defined(_WIN32) || defined(_WIN64)
#define EMBEDLET_WINDOWS 1
#define EMBEDLET_POSIX 0
#else
#define EMBEDLET_WINDOWS 0
#define EMBEDLET_POSIX 1
#endif

/*============================================================================
 * Includes
 *============================================================================*/

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if EMBEDLET_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Optional SSE2 acceleration */
#if defined(__SSE2__) ||                                                       \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <emmintrin.h>
#define EMBEDLET_HAS_SSE2 1
#else
#define EMBEDLET_HAS_SSE2 0
#endif

/*============================================================================
 * Error Codes
 *============================================================================*/

#define EMBEDLET_OK 0
#define EMBEDLET_ERR_INVALID_ARG -1
#define EMBEDLET_ERR_INVALID_ID -2
#define EMBEDLET_ERR_FILE_OPEN -3
#define EMBEDLET_ERR_MMAP -4
#define EMBEDLET_ERR_ALLOC -5
#define EMBEDLET_ERR_TRUNCATE -6
#define EMBEDLET_ERR_THREAD -7
#define EMBEDLET_ERR_NOT_FOUND -8

/*============================================================================
 * Constants
 *============================================================================*/

#define EMBEDLET_AUTO_THREADS 0
#define EMBEDLET_SINGLE_THREAD 1

/*============================================================================
 * Types
 *============================================================================*/

/**
 * @brief Result structure for top-N similarity queries.
 */
typedef struct embedlet_result {
  size_t id;   /**< Embedding index */
  float score; /**< Similarity score (cosine similarity) */
} embedlet_result_t;

/**
 * @brief Opaque handle to an embedding store.
 */
typedef struct embedlet_store embedlet_store_t;

/*============================================================================
 * Public API Declarations
 *============================================================================*/

/**
 * @brief Open or create an embedding store.
 * @param path      File path for the store.
 * @param dims      Dimensionality of embeddings (must be > 0).
 * @param store_out Pointer to receive the store handle.
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_open(const char *path, size_t dims, embedlet_store_t **store_out);

/**
 * @brief Close an embedding store and optionally compact (trim trailing zeros).
 * @param store   Store handle.
 * @param compact If true, truncate trailing zeroed embeddings.
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_close(embedlet_store_t *store, bool compact);

/**
 * @brief Get the current number of embeddings in the store.
 * @param store Store handle.
 * @return Number of embeddings (file_size / (dims * sizeof(float))).
 */
size_t embedlet_count(const embedlet_store_t *store);

/**
 * @brief Append a new embedding to the store.
 * @param store  Store handle.
 * @param data   Pointer to dims floats.
 * @param reuse  If true, reuse first zeroed slot; otherwise always append.
 * @param id_out Pointer to receive the assigned index.
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_append(embedlet_store_t *store, const float *data, bool reuse,
                    size_t *id_out);

/**
 * @brief Replace the embedding at a given index.
 * @param store Store handle.
 * @param id    Index to replace.
 * @param data  Pointer to dims floats.
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_replace(embedlet_store_t *store, size_t id, const float *data);

/**
 * @brief Delete an embedding by zeroing its data.
 * @param store Store handle.
 * @param id    Index to delete.
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_delete(embedlet_store_t *store, size_t id);

/**
 * @brief Get a pointer to the embedding at a given index (read-only).
 * @param store Store handle.
 * @param id    Index to retrieve.
 * @return Pointer to dims floats, or NULL if id is invalid.
 */
const float *embedlet_get(const embedlet_store_t *store, size_t id);

/**
 * @brief Check if an embedding slot is zeroed (deleted).
 * @param store Store handle.
 * @param id    Index to check.
 * @return true if zeroed, false otherwise.
 */
bool embedlet_is_zeroed(const embedlet_store_t *store, size_t id);

/**
 * @brief Compute cosine similarity between two embeddings.
 * @param store Store handle (for dims).
 * @param a     First embedding (dims floats).
 * @param b     Second embedding (dims floats).
 * @return Cosine similarity in [-1, 1].
 */
float embedlet_similarity(const embedlet_store_t *store, const float *a,
                          const float *b);

/**
 * @brief Compute cosine similarity between two raw float arrays.
 * @param a    First array.
 * @param b    Second array.
 * @param dims Number of dimensions.
 * @return Cosine similarity in [-1, 1].
 */
float embedlet_similarity_raw(const float *a, const float *b, size_t dims);

/**
 * @brief Find the top-N most or least similar embeddings.
 * @param store        Store handle.
 * @param query        Query embedding (dims floats).
 * @param n            Number of results to return.
 * @param most_similar If true, return most similar; if false, least similar.
 * @param num_threads  Thread count: EMBEDLET_AUTO_THREADS,
 *                     EMBEDLET_SINGLE_THREAD, or specific count.
 * @param results      Array of n embedlet_result_t to receive results (sorted
 *                     by score).
 * @param count_out    Pointer to receive actual number of results (may be < n).
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_search(embedlet_store_t *store, const float *query, size_t n,
                    bool most_similar, int num_threads,
                    embedlet_result_t *results, size_t *count_out);

/**
 * @brief Explicitly compact the store (truncate trailing zeros).
 * @param store Store handle.
 * @return EMBEDLET_OK on success, error code otherwise.
 */
int embedlet_compact(embedlet_store_t *store);

/**
 * @brief Get the dimensionality of the store.
 * @param store Store handle.
 * @return Number of dimensions.
 */
size_t embedlet_dims(const embedlet_store_t *store);

/*============================================================================
 * Implementation
 *============================================================================*/

#ifdef EMBEDLET_IMPLEMENTATION

/*----------------------------------------------------------------------------
 * Internal Types and Structures
 *----------------------------------------------------------------------------*/

/* Platform-specific mutex type */
#if EMBEDLET_WINDOWS
typedef CRITICAL_SECTION embedlet_mutex_t;
#else
typedef pthread_mutex_t embedlet_mutex_t;
#endif

typedef struct embedlet_work {
  void (*func)(void *arg);
  void *arg;
  struct embedlet_work *next;
} embedlet_work_t;

typedef struct embedlet_pool {
  int num_threads;
  volatile bool shutdown;
  volatile int active_count;
  volatile int pending_count;
  embedlet_work_t *work_head;
  embedlet_work_t *work_tail;
#if EMBEDLET_WINDOWS
  HANDLE *threads;
  CRITICAL_SECTION mutex;
  CONDITION_VARIABLE cond_work;
  CONDITION_VARIABLE cond_done;
#else
  pthread_t *threads;
  pthread_mutex_t mutex;
  pthread_cond_t cond_work;
  pthread_cond_t cond_done;
#endif
} embedlet_pool_t;

struct embedlet_store {
  size_t dims;
  size_t capacity;
  size_t file_size;
  float *data;
  char *path;
  embedlet_mutex_t mutex;
  embedlet_pool_t *pool;
#if EMBEDLET_WINDOWS
  HANDLE file_handle;
  HANDLE map_handle;
#else
  int fd;
#endif
};

typedef struct {
  const embedlet_store_t *store;
  const float *query;
  float query_norm;
  size_t start;
  size_t end;
  embedlet_result_t *local_results;
  size_t n;
  bool most_similar;
  size_t result_count;
} embedlet_search_task_t;

/*----------------------------------------------------------------------------
 * Platform-Specific Mutex Operations
 *----------------------------------------------------------------------------*/

static inline void embedlet_mutex_init(embedlet_mutex_t *m) {
#if EMBEDLET_WINDOWS
  InitializeCriticalSection(m);
#else
  pthread_mutex_init(m, NULL);
#endif
}

static inline void embedlet_mutex_destroy(embedlet_mutex_t *m) {
#if EMBEDLET_WINDOWS
  DeleteCriticalSection(m);
#else
  pthread_mutex_destroy(m);
#endif
}

static inline void embedlet_mutex_lock(embedlet_mutex_t *m) {
#if EMBEDLET_WINDOWS
  EnterCriticalSection(m);
#else
  pthread_mutex_lock(m);
#endif
}

static inline void embedlet_mutex_unlock(embedlet_mutex_t *m) {
#if EMBEDLET_WINDOWS
  LeaveCriticalSection(m);
#else
  pthread_mutex_unlock(m);
#endif
}

/*----------------------------------------------------------------------------
 * Thread Pool Implementation
 *----------------------------------------------------------------------------*/

#if EMBEDLET_WINDOWS
static DWORD WINAPI embedlet_pool_worker(LPVOID arg) {
#else
static void *embedlet_pool_worker(void *arg) {
#endif
  embedlet_pool_t *pool = (embedlet_pool_t *)arg;

  for (;;) {
#if EMBEDLET_WINDOWS
    EnterCriticalSection(&pool->mutex);
    while (!pool->shutdown && pool->work_head == NULL) {
      SleepConditionVariableCS(&pool->cond_work, &pool->mutex, INFINITE);
    }
#else
    pthread_mutex_lock(&pool->mutex);
    while (!pool->shutdown && pool->work_head == NULL) {
      pthread_cond_wait(&pool->cond_work, &pool->mutex);
    }
#endif
    if (pool->shutdown && pool->work_head == NULL) {
#if EMBEDLET_WINDOWS
      LeaveCriticalSection(&pool->mutex);
      return 0;
#else
      pthread_mutex_unlock(&pool->mutex);
      return NULL;
#endif
    }

    embedlet_work_t *work = pool->work_head;
    if (work) {
      pool->work_head = work->next;
      if (!pool->work_head)
        pool->work_tail = NULL;
      pool->active_count++;
    }
#if EMBEDLET_WINDOWS
    LeaveCriticalSection(&pool->mutex);
#else
    pthread_mutex_unlock(&pool->mutex);
#endif

    if (work) {
      work->func(work->arg);
      free(work);

#if EMBEDLET_WINDOWS
      EnterCriticalSection(&pool->mutex);
      pool->active_count--;
      pool->pending_count--;
      if (pool->pending_count == 0 && pool->active_count == 0) {
        WakeAllConditionVariable(&pool->cond_done);
      }
      LeaveCriticalSection(&pool->mutex);
#else
      pthread_mutex_lock(&pool->mutex);
      pool->active_count--;
      pool->pending_count--;
      if (pool->pending_count == 0 && pool->active_count == 0) {
        pthread_cond_broadcast(&pool->cond_done);
      }
      pthread_mutex_unlock(&pool->mutex);
#endif
    }
  }
}

static embedlet_pool_t *embedlet_pool_create(int num_threads) {
  if (num_threads <= 0)
    return NULL;

  embedlet_pool_t *pool = (embedlet_pool_t *)calloc(1, sizeof(embedlet_pool_t));
  if (!pool)
    return NULL;

  pool->num_threads = num_threads;
  pool->shutdown = false;
  pool->active_count = 0;
  pool->pending_count = 0;
  pool->work_head = NULL;
  pool->work_tail = NULL;

#if EMBEDLET_WINDOWS
  InitializeCriticalSection(&pool->mutex);
  InitializeConditionVariable(&pool->cond_work);
  InitializeConditionVariable(&pool->cond_done);
  pool->threads = (HANDLE *)calloc((size_t)num_threads, sizeof(HANDLE));
  if (!pool->threads) {
    DeleteCriticalSection(&pool->mutex);
    free(pool);
    return NULL;
  }

  int created = 0;
  for (int i = 0; i < num_threads; i++) {
    pool->threads[i] =
        CreateThread(NULL, 0, embedlet_pool_worker, pool, 0, NULL);
    if (pool->threads[i] == NULL) {
      EnterCriticalSection(&pool->mutex);
      pool->shutdown = true;
      WakeAllConditionVariable(&pool->cond_work);
      LeaveCriticalSection(&pool->mutex);

      for (int j = 0; j < created; j++) {
        WaitForSingleObject(pool->threads[j], INFINITE);
        CloseHandle(pool->threads[j]);
      }

      DeleteCriticalSection(&pool->mutex);
      free(pool->threads);
      free(pool);
      return NULL;
    }
    created++;
  }
#else
  pthread_mutex_init(&pool->mutex, NULL);
  pthread_cond_init(&pool->cond_work, NULL);
  pthread_cond_init(&pool->cond_done, NULL);
  pool->threads = (pthread_t *)calloc((size_t)num_threads, sizeof(pthread_t));
  if (!pool->threads) {
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_work);
    pthread_cond_destroy(&pool->cond_done);
    free(pool);
    return NULL;
  }
  for (int i = 0; i < num_threads; i++) {
    int ret =
        pthread_create(&pool->threads[i], NULL, embedlet_pool_worker, pool);
    if (ret != 0) {
      pthread_mutex_lock(&pool->mutex);
      pool->shutdown = true;
      pthread_cond_broadcast(&pool->cond_work);
      pthread_mutex_unlock(&pool->mutex);

      for (int j = 0; j < i; j++) {
        pthread_join(pool->threads[j], NULL);
      }
      pthread_mutex_destroy(&pool->mutex);
      pthread_cond_destroy(&pool->cond_work);
      pthread_cond_destroy(&pool->cond_done);
      free(pool->threads);
      free(pool);
      return NULL;
    }
  }
#endif

  return pool;
}

static void embedlet_pool_submit(embedlet_pool_t *pool, void (*func)(void *),
                                 void *arg) {
  embedlet_work_t *work = (embedlet_work_t *)malloc(sizeof(embedlet_work_t));
  if (!work)
    return;

  work->func = func;
  work->arg = arg;
  work->next = NULL;

#if EMBEDLET_WINDOWS
  EnterCriticalSection(&pool->mutex);
  if (pool->work_tail) {
    pool->work_tail->next = work;
  } else {
    pool->work_head = work;
  }
  pool->work_tail = work;
  pool->pending_count++;
  WakeConditionVariable(&pool->cond_work);
  LeaveCriticalSection(&pool->mutex);
#else
  pthread_mutex_lock(&pool->mutex);
  if (pool->work_tail) {
    pool->work_tail->next = work;
  } else {
    pool->work_head = work;
  }
  pool->work_tail = work;
  pool->pending_count++;
  pthread_cond_signal(&pool->cond_work);
  pthread_mutex_unlock(&pool->mutex);
#endif
}

static void embedlet_pool_wait(embedlet_pool_t *pool) {
#if EMBEDLET_WINDOWS
  EnterCriticalSection(&pool->mutex);
  while (pool->pending_count > 0 || pool->active_count > 0) {
    SleepConditionVariableCS(&pool->cond_done, &pool->mutex, INFINITE);
  }
  LeaveCriticalSection(&pool->mutex);
#else
  pthread_mutex_lock(&pool->mutex);
  while (pool->pending_count > 0 || pool->active_count > 0) {
    pthread_cond_wait(&pool->cond_done, &pool->mutex);
  }
  pthread_mutex_unlock(&pool->mutex);
#endif
}

static void embedlet_pool_destroy(embedlet_pool_t *pool) {
  if (!pool)
    return;

#if EMBEDLET_WINDOWS
  EnterCriticalSection(&pool->mutex);
  pool->shutdown = true;
  WakeAllConditionVariable(&pool->cond_work);
  LeaveCriticalSection(&pool->mutex);

  for (int i = 0; i < pool->num_threads; i++) {
    WaitForSingleObject(pool->threads[i], INFINITE);
    CloseHandle(pool->threads[i]);
  }
  DeleteCriticalSection(&pool->mutex);
#else
  pthread_mutex_lock(&pool->mutex);
  pool->shutdown = true;
  pthread_cond_broadcast(&pool->cond_work);
  pthread_mutex_unlock(&pool->mutex);

  for (int i = 0; i < pool->num_threads; i++) {
    pthread_join(pool->threads[i], NULL);
  }
  pthread_mutex_destroy(&pool->mutex);
  pthread_cond_destroy(&pool->cond_work);
  pthread_cond_destroy(&pool->cond_done);
#endif

  embedlet_work_t *work = pool->work_head;
  while (work) {
    embedlet_work_t *next = work->next;
    free(work);
    work = next;
  }

  free(pool->threads);
  free(pool);
}

static int embedlet_get_cpu_count(void) {
#if EMBEDLET_WINDOWS
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return (int)sysinfo.dwNumberOfProcessors;
#else
  long count = sysconf(_SC_NPROCESSORS_ONLN);
  return count > 0 ? (int)count : 1;
#endif
}

/*----------------------------------------------------------------------------
 * Similarity Functions (C + optional SSE2)
 *----------------------------------------------------------------------------*/

#if EMBEDLET_HAS_SSE2

static float embedlet_hsum_sse(__m128 v) {
  __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

static float embedlet_dot_sse2(const float *a, const float *b, size_t n) {
  __m128 sum = _mm_setzero_ps();
  size_t i = 0;

  for (; i + 4 <= n; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
  }

  float result = embedlet_hsum_sse(sum);

  for (; i < n; i++) {
    result += a[i] * b[i];
  }

  return result;
}

static float embedlet_norm_sse2(const float *a, size_t n) {
  __m128 sum = _mm_setzero_ps();
  size_t i = 0;

  for (; i + 4 <= n; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    sum = _mm_add_ps(sum, _mm_mul_ps(va, va));
  }

  float result = embedlet_hsum_sse(sum);

  for (; i < n; i++) {
    result += a[i] * a[i];
  }

  return sqrtf(result);
}

#endif /* EMBEDLET_HAS_SSE2 */

static float embedlet_dot_c(const float *a, const float *b, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static float embedlet_norm_c(const float *a, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    sum += a[i] * a[i];
  }
  return sqrtf(sum);
}

static float embedlet_dot(const float *a, const float *b, size_t n) {
#if EMBEDLET_HAS_SSE2
  return embedlet_dot_sse2(a, b, n);
#else
  return embedlet_dot_c(a, b, n);
#endif
}

static float embedlet_norm(const float *a, size_t n) {
#if EMBEDLET_HAS_SSE2
  return embedlet_norm_sse2(a, n);
#else
  return embedlet_norm_c(a, n);
#endif
}

/*----------------------------------------------------------------------------
 * Platform-Specific File/Mmap Operations
 *----------------------------------------------------------------------------*/

#if EMBEDLET_WINDOWS

static int embedlet_file_open(embedlet_store_t *store, const char *path) {
  store->file_handle =
      CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL,
                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (store->file_handle == INVALID_HANDLE_VALUE) {
    return EMBEDLET_ERR_FILE_OPEN;
  }

  LARGE_INTEGER size;
  if (!GetFileSizeEx(store->file_handle, &size)) {
    CloseHandle(store->file_handle);
    return EMBEDLET_ERR_FILE_OPEN;
  }
  store->file_size = (size_t)size.QuadPart;

  return EMBEDLET_OK;
}

static int embedlet_mmap_update(embedlet_store_t *store, size_t new_capacity) {
  if (store->map_handle) {
    if (store->data) {
      UnmapViewOfFile(store->data);
      store->data = NULL;
    }
    CloseHandle(store->map_handle);
    store->map_handle = NULL;
  }

  if (new_capacity == 0) {
    store->capacity = 0;
    return EMBEDLET_OK;
  }

  LARGE_INTEGER li;
  li.QuadPart = (LONGLONG)new_capacity;

  store->map_handle = CreateFileMappingA(
      store->file_handle, NULL, PAGE_READWRITE, li.HighPart, li.LowPart, NULL);
  if (!store->map_handle) {
    return EMBEDLET_ERR_MMAP;
  }

  store->data = (float *)MapViewOfFile(store->map_handle, FILE_MAP_ALL_ACCESS,
                                       0, 0, new_capacity);
  if (!store->data) {
    CloseHandle(store->map_handle);
    store->map_handle = NULL;
    return EMBEDLET_ERR_MMAP;
  }

  store->capacity = new_capacity;
  return EMBEDLET_OK;
}

static int embedlet_file_resize(embedlet_store_t *store, size_t new_size) {
  if (store->map_handle) {
    if (store->data) {
      UnmapViewOfFile(store->data);
      store->data = NULL;
    }
    CloseHandle(store->map_handle);
    store->map_handle = NULL;
  }

  LARGE_INTEGER li;
  li.QuadPart = (LONGLONG)new_size;
  if (!SetFilePointerEx(store->file_handle, li, NULL, FILE_BEGIN)) {
    return EMBEDLET_ERR_TRUNCATE;
  }
  if (!SetEndOfFile(store->file_handle)) {
    return EMBEDLET_ERR_TRUNCATE;
  }

  store->file_size = new_size;
  return EMBEDLET_OK;
}

static void embedlet_file_close(embedlet_store_t *store) {
  if (store->data) {
    UnmapViewOfFile(store->data);
    store->data = NULL;
  }
  if (store->map_handle) {
    CloseHandle(store->map_handle);
    store->map_handle = NULL;
  }
  if (store->file_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(store->file_handle);
    store->file_handle = INVALID_HANDLE_VALUE;
  }
}

#else /* POSIX */

static int embedlet_file_open(embedlet_store_t *store, const char *path) {
  store->fd = open(path, O_RDWR | O_CREAT, 0644);
  if (store->fd < 0) {
    return EMBEDLET_ERR_FILE_OPEN;
  }

  struct stat st;
  if (fstat(store->fd, &st) < 0) {
    close(store->fd);
    return EMBEDLET_ERR_FILE_OPEN;
  }
  store->file_size = (size_t)st.st_size;

  return EMBEDLET_OK;
}

static int embedlet_mmap_update(embedlet_store_t *store, size_t new_capacity) {
  if (store->data && store->capacity > 0) {
    munmap(store->data, store->capacity);
    store->data = NULL;
  }

  if (new_capacity == 0) {
    store->capacity = 0;
    return EMBEDLET_OK;
  }

  store->data = (float *)mmap(NULL, new_capacity, PROT_READ | PROT_WRITE,
                              MAP_SHARED, store->fd, 0);
  if (store->data == MAP_FAILED) {
    store->data = NULL;
    return EMBEDLET_ERR_MMAP;
  }

  store->capacity = new_capacity;
  return EMBEDLET_OK;
}

static int embedlet_file_resize(embedlet_store_t *store, size_t new_size) {
  if (store->data && store->capacity > 0) {
    munmap(store->data, store->capacity);
    store->data = NULL;
    store->capacity = 0;
  }

  if (ftruncate(store->fd, (off_t)new_size) < 0) {
    return EMBEDLET_ERR_TRUNCATE;
  }

  store->file_size = new_size;
  return EMBEDLET_OK;
}

static void embedlet_file_close(embedlet_store_t *store) {
  if (store->data && store->capacity > 0) {
    munmap(store->data, store->capacity);
    store->data = NULL;
  }
  if (store->fd >= 0) {
    close(store->fd);
    store->fd = -1;
  }
}

#endif /* EMBEDLET_WINDOWS/POSIX */

/*----------------------------------------------------------------------------
 * Helper Functions
 *----------------------------------------------------------------------------*/

static size_t embedlet_embedding_size(const embedlet_store_t *store) {
  return store->dims * sizeof(float);
}

static bool embedlet_is_zeroed_ptr(const float *data, size_t dims) {
  for (size_t i = 0; i < dims; i++) {
    if (data[i] != 0.0f)
      return false;
  }
  return true;
}

static int embedlet_ensure_capacity(embedlet_store_t *store,
                                    size_t needed_bytes) {
  if (needed_bytes <= store->capacity) {
    return EMBEDLET_OK;
  }

  size_t new_cap = store->capacity ? store->capacity * 2 : 4096;
  while (new_cap < needed_bytes) {
    new_cap *= 2;
  }

  int err = embedlet_file_resize(store, new_cap);
  if (err != EMBEDLET_OK)
    return err;

  err = embedlet_mmap_update(store, new_cap);
  if (err != EMBEDLET_OK)
    return err;

  return EMBEDLET_OK;
}

/*----------------------------------------------------------------------------
 * Min-Heap for Top-N (most similar)
 *----------------------------------------------------------------------------*/

static void embedlet_heap_push_min(embedlet_result_t *heap, size_t *size,
                                   size_t max_size, size_t id, float score) {
  if (*size < max_size) {
    heap[*size].id = id;
    heap[*size].score = score;
    (*size)++;
    size_t i = *size - 1;
    while (i > 0) {
      size_t parent = (i - 1) / 2;
      if (heap[parent].score <= heap[i].score)
        break;
      embedlet_result_t tmp = heap[parent];
      heap[parent] = heap[i];
      heap[i] = tmp;
      i = parent;
    }
  } else if (score > heap[0].score) {
    heap[0].id = id;
    heap[0].score = score;
    size_t i = 0;
    for (;;) {
      size_t left = 2 * i + 1;
      size_t right = 2 * i + 2;
      size_t smallest = i;
      if (left < max_size && heap[left].score < heap[smallest].score)
        smallest = left;
      if (right < max_size && heap[right].score < heap[smallest].score)
        smallest = right;
      if (smallest == i)
        break;
      embedlet_result_t tmp = heap[i];
      heap[i] = heap[smallest];
      heap[smallest] = tmp;
      i = smallest;
    }
  }
}

/*----------------------------------------------------------------------------
 * Max-Heap for Top-N (least similar)
 *----------------------------------------------------------------------------*/

static void embedlet_heap_push_max(embedlet_result_t *heap, size_t *size,
                                   size_t max_size, size_t id, float score) {
  if (*size < max_size) {
    heap[*size].id = id;
    heap[*size].score = score;
    (*size)++;
    size_t i = *size - 1;
    while (i > 0) {
      size_t parent = (i - 1) / 2;
      if (heap[parent].score >= heap[i].score)
        break;
      embedlet_result_t tmp = heap[parent];
      heap[parent] = heap[i];
      heap[i] = tmp;
      i = parent;
    }
  } else if (score < heap[0].score) {
    heap[0].id = id;
    heap[0].score = score;
    size_t i = 0;
    for (;;) {
      size_t left = 2 * i + 1;
      size_t right = 2 * i + 2;
      size_t largest = i;
      if (left < max_size && heap[left].score > heap[largest].score)
        largest = left;
      if (right < max_size && heap[right].score > heap[largest].score)
        largest = right;
      if (largest == i)
        break;
      embedlet_result_t tmp = heap[i];
      heap[i] = heap[largest];
      heap[largest] = tmp;
      i = largest;
    }
  }
}

static int embedlet_cmp_desc(const void *a, const void *b) {
  float sa = ((const embedlet_result_t *)a)->score;
  float sb = ((const embedlet_result_t *)b)->score;
  return (sa < sb) ? 1 : (sa > sb) ? -1 : 0;
}

static int embedlet_cmp_asc(const void *a, const void *b) {
  float sa = ((const embedlet_result_t *)a)->score;
  float sb = ((const embedlet_result_t *)b)->score;
  return (sa > sb) ? 1 : (sa < sb) ? -1 : 0;
}

/*----------------------------------------------------------------------------
 * Search Task Worker
 *----------------------------------------------------------------------------*/

static void embedlet_search_worker(void *arg) {
  embedlet_search_task_t *task = (embedlet_search_task_t *)arg;
  const embedlet_store_t *store = task->store;
  const float *query = task->query;
  float query_norm = task->query_norm;
  size_t dims = store->dims;

  size_t heap_size = 0;

  for (size_t i = task->start; i < task->end; i++) {
    const float *emb = store->data + i * dims;

    if (embedlet_is_zeroed_ptr(emb, dims))
      continue;

    float dot = embedlet_dot(query, emb, dims);
    float emb_norm = embedlet_norm(emb, dims);
    float sim = (query_norm > 0.0f && emb_norm > 0.0f)
                    ? dot / (query_norm * emb_norm)
                    : 0.0f;

    if (task->most_similar) {
      embedlet_heap_push_min(task->local_results, &heap_size, task->n, i, sim);
    } else {
      embedlet_heap_push_max(task->local_results, &heap_size, task->n, i, sim);
    }
  }

  task->result_count = heap_size;
}

/*----------------------------------------------------------------------------
 * Public API Implementation
 *----------------------------------------------------------------------------*/

int embedlet_open(const char *path, size_t dims, embedlet_store_t **store_out) {
  if (!path || dims == 0 || !store_out) {
    return EMBEDLET_ERR_INVALID_ARG;
  }

  embedlet_store_t *store =
      (embedlet_store_t *)calloc(1, sizeof(embedlet_store_t));
  if (!store)
    return EMBEDLET_ERR_ALLOC;

  store->dims = dims;
  store->path = strdup(path);
  if (!store->path) {
    free(store);
    return EMBEDLET_ERR_ALLOC;
  }

#if EMBEDLET_WINDOWS
  store->file_handle = INVALID_HANDLE_VALUE;
  store->map_handle = NULL;
#else
  store->fd = -1;
#endif
  store->data = NULL;
  store->capacity = 0;
  store->pool = NULL;

  embedlet_mutex_init(&store->mutex);

  int err = embedlet_file_open(store, path);
  if (err != EMBEDLET_OK) {
    embedlet_mutex_destroy(&store->mutex);
    free(store->path);
    free(store);
    return err;
  }

  if (store->file_size > 0) {
    err = embedlet_mmap_update(store, store->file_size);
    if (err != EMBEDLET_OK) {
      embedlet_file_close(store);
      embedlet_mutex_destroy(&store->mutex);
      free(store->path);
      free(store);
      return err;
    }
  }

  *store_out = store;
  return EMBEDLET_OK;
}

int embedlet_close(embedlet_store_t *store, bool compact) {
  if (!store)
    return EMBEDLET_ERR_INVALID_ARG;

  if (compact) {
    embedlet_compact(store);
  }

  if (store->pool) {
    embedlet_pool_destroy(store->pool);
    store->pool = NULL;
  }

  embedlet_file_close(store);
  embedlet_mutex_destroy(&store->mutex);
  free(store->path);
  free(store);

  return EMBEDLET_OK;
}

size_t embedlet_count(const embedlet_store_t *store) {
  if (!store || store->dims == 0)
    return 0;
  return store->file_size / (store->dims * sizeof(float));
}

size_t embedlet_dims(const embedlet_store_t *store) {
  return store ? store->dims : 0;
}

int embedlet_append(embedlet_store_t *store, const float *data, bool reuse,
                    size_t *id_out) {
  if (!store || !data || !id_out) {
    return EMBEDLET_ERR_INVALID_ARG;
  }

  embedlet_mutex_lock(&store->mutex);

  size_t emb_size = embedlet_embedding_size(store);
  size_t count = store->file_size / emb_size;
  size_t target_id = count;

  if (reuse && store->data) {
    for (size_t i = 0; i < count; i++) {
      if (embedlet_is_zeroed_ptr(store->data + i * store->dims, store->dims)) {
        target_id = i;
        break;
      }
    }
  }

  if (target_id == count) {
    size_t new_file_size = store->file_size + emb_size;
    int err = embedlet_ensure_capacity(store, new_file_size);
    if (err != EMBEDLET_OK) {
      embedlet_mutex_unlock(&store->mutex);
      return err;
    }
    store->file_size = new_file_size;
  }

  memcpy(store->data + target_id * store->dims, data, emb_size);

  *id_out = target_id;

  embedlet_mutex_unlock(&store->mutex);
  return EMBEDLET_OK;
}

int embedlet_replace(embedlet_store_t *store, size_t id, const float *data) {
  if (!store || !data) {
    return EMBEDLET_ERR_INVALID_ARG;
  }

  embedlet_mutex_lock(&store->mutex);

  size_t count = embedlet_count(store);
  if (id >= count) {
    embedlet_mutex_unlock(&store->mutex);
    return EMBEDLET_ERR_INVALID_ID;
  }

  memcpy(store->data + id * store->dims, data, embedlet_embedding_size(store));

  embedlet_mutex_unlock(&store->mutex);
  return EMBEDLET_OK;
}

int embedlet_delete(embedlet_store_t *store, size_t id) {
  if (!store)
    return EMBEDLET_ERR_INVALID_ARG;

  embedlet_mutex_lock(&store->mutex);

  size_t count = embedlet_count(store);
  if (id >= count) {
    embedlet_mutex_unlock(&store->mutex);
    return EMBEDLET_ERR_INVALID_ID;
  }

  memset(store->data + id * store->dims, 0, embedlet_embedding_size(store));

  embedlet_mutex_unlock(&store->mutex);
  return EMBEDLET_OK;
}

const float *embedlet_get(const embedlet_store_t *store, size_t id) {
  if (!store || !store->data)
    return NULL;
  size_t count = embedlet_count(store);
  if (id >= count)
    return NULL;
  return store->data + id * store->dims;
}

bool embedlet_is_zeroed(const embedlet_store_t *store, size_t id) {
  const float *emb = embedlet_get(store, id);
  if (!emb)
    return true;
  return embedlet_is_zeroed_ptr(emb, store->dims);
}

float embedlet_similarity(const embedlet_store_t *store, const float *a,
                          const float *b) {
  if (!store || !a || !b)
    return 0.0f;
  return embedlet_similarity_raw(a, b, store->dims);
}

float embedlet_similarity_raw(const float *a, const float *b, size_t dims) {
  if (!a || !b || dims == 0)
    return 0.0f;

  float dot = embedlet_dot(a, b, dims);
  float na = embedlet_norm(a, dims);
  float nb = embedlet_norm(b, dims);

  if (na < FLT_EPSILON || nb < FLT_EPSILON)
    return 0.0f;
  return dot / (na * nb);
}

int embedlet_compact(embedlet_store_t *store) {
  if (!store)
    return EMBEDLET_ERR_INVALID_ARG;

  embedlet_mutex_lock(&store->mutex);

  size_t count = embedlet_count(store);
  if (count == 0) {
    embedlet_mutex_unlock(&store->mutex);
    return EMBEDLET_OK;
  }

  size_t last_nonzero = count;
  while (last_nonzero > 0) {
    if (!embedlet_is_zeroed_ptr(store->data + (last_nonzero - 1) * store->dims,
                                store->dims)) {
      break;
    }
    last_nonzero--;
  }

  if (last_nonzero < count) {
    size_t new_size = last_nonzero * embedlet_embedding_size(store);
    int err = embedlet_file_resize(store, new_size);
    if (err != EMBEDLET_OK) {
      embedlet_mutex_unlock(&store->mutex);
      return err;
    }

    if (new_size > 0) {
      err = embedlet_mmap_update(store, new_size);
      if (err != EMBEDLET_OK) {
        embedlet_mutex_unlock(&store->mutex);
        return err;
      }
    }
  }

  embedlet_mutex_unlock(&store->mutex);
  return EMBEDLET_OK;
}

int embedlet_search(embedlet_store_t *store, const float *query, size_t n,
                    bool most_similar, int num_threads,
                    embedlet_result_t *results, size_t *count_out) {
  if (!store || !query || n == 0 || !results || !count_out) {
    return EMBEDLET_ERR_INVALID_ARG;
  }

  size_t total = embedlet_count(store);
  if (total == 0) {
    *count_out = 0;
    return EMBEDLET_OK;
  }

  float query_norm = embedlet_norm(query, store->dims);

  int threads = num_threads;
  if (threads == EMBEDLET_AUTO_THREADS) {
    threads = embedlet_get_cpu_count();
    if (threads > 8)
      threads = 8;
  }
  if (threads < 1)
    threads = 1;
  if ((size_t)threads > total)
    threads = (int)total;

  if (threads == 1) {
    size_t heap_size = 0;
    for (size_t i = 0; i < total; i++) {
      const float *emb = store->data + i * store->dims;
      if (embedlet_is_zeroed_ptr(emb, store->dims))
        continue;

      float dot = embedlet_dot(query, emb, store->dims);
      float emb_norm = embedlet_norm(emb, store->dims);
      float sim = (query_norm > FLT_EPSILON && emb_norm > FLT_EPSILON)
                      ? dot / (query_norm * emb_norm)
                      : 0.0f;

      if (most_similar) {
        embedlet_heap_push_min(results, &heap_size, n, i, sim);
      } else {
        embedlet_heap_push_max(results, &heap_size, n, i, sim);
      }
    }

    qsort(results, heap_size, sizeof(embedlet_result_t),
          most_similar ? embedlet_cmp_desc : embedlet_cmp_asc);

    *count_out = heap_size;
    return EMBEDLET_OK;
  }

  embedlet_mutex_lock(&store->mutex);
  if (!store->pool) {
    store->pool = embedlet_pool_create(threads);
    if (!store->pool) {
      embedlet_mutex_unlock(&store->mutex);
      return EMBEDLET_ERR_THREAD;
    }
  }
  embedlet_pool_t *pool = store->pool;
  embedlet_mutex_unlock(&store->mutex);

  embedlet_search_task_t *tasks = (embedlet_search_task_t *)calloc(
      (size_t)threads, sizeof(embedlet_search_task_t));
  if (!tasks)
    return EMBEDLET_ERR_ALLOC;

  size_t chunk = total / (size_t)threads;
  size_t remainder = total % (size_t)threads;

  for (int i = 0; i < threads; i++) {
    tasks[i].store = store;
    tasks[i].query = query;
    tasks[i].query_norm = query_norm;
    tasks[i].n = n;
    tasks[i].most_similar = most_similar;
    tasks[i].start =
        (size_t)i * chunk + ((size_t)i < remainder ? (size_t)i : remainder);
    tasks[i].end = tasks[i].start + chunk + ((size_t)i < remainder ? 1 : 0);
    tasks[i].local_results =
        (embedlet_result_t *)calloc(n, sizeof(embedlet_result_t));
    tasks[i].result_count = 0;

    if (!tasks[i].local_results) {
      for (int j = 0; j < i; j++)
        free(tasks[j].local_results);
      free(tasks);
      return EMBEDLET_ERR_ALLOC;
    }

    embedlet_pool_submit(pool, embedlet_search_worker, &tasks[i]);
  }

  embedlet_pool_wait(pool);

  size_t final_heap_size = 0;
  for (int i = 0; i < threads; i++) {
    for (size_t j = 0; j < tasks[i].result_count; j++) {
      if (most_similar) {
        embedlet_heap_push_min(results, &final_heap_size, n,
                               tasks[i].local_results[j].id,
                               tasks[i].local_results[j].score);
      } else {
        embedlet_heap_push_max(results, &final_heap_size, n,
                               tasks[i].local_results[j].id,
                               tasks[i].local_results[j].score);
      }
    }
    free(tasks[i].local_results);
  }
  free(tasks);

  qsort(results, final_heap_size, sizeof(embedlet_result_t),
        most_similar ? embedlet_cmp_desc : embedlet_cmp_asc);

  *count_out = final_heap_size;
  return EMBEDLET_OK;
}

#endif /* EMBEDLET_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif

#endif /* EMBEDLET_H */
