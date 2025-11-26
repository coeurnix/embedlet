"""
example_equivalent.py

This script provides a Python equivalent of examples/example.c, demonstrating
similar operations on embeddings using NumPy for vector computations and
cosine similarity, to compare outputs with the embedlet C library.
Embeddings from *.dat are used with the Qwen/Qwen3-Embedding-0.6B model.
"""

import numpy as np
import struct
import os

DIMS = 1024

class EmbedletStore:
    def __init__(self, dims):
        self.dims = dims
        self.embeddings = []  # List of np arrays, None for deleted

    def load_embedding(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            return np.array(struct.unpack('<1024f', data), dtype=np.float32)

    def count(self):
        return len([e for e in self.embeddings if e is not None])

    def append(self, emb, reuse):
        if reuse:
            for i, e in enumerate(self.embeddings):
                if e is None:
                    self.embeddings[i] = emb
                    return i
        self.embeddings.append(emb)
        return len(self.embeddings) - 1

    def delete(self, idx):
        if 0 <= idx < len(self.embeddings):
            self.embeddings[idx] = None

    def is_zeroed(self, idx):
        return self.embeddings[idx] is None

    def replace(self, idx, emb):
        if 0 <= idx < len(self.embeddings):
            self.embeddings[idx] = emb

    def compact(self):
        while self.embeddings and self.embeddings[-1] is None:
            self.embeddings.pop()
        new_embeddings = []
        for emb in self.embeddings:
            if emb is not None:
                new_embeddings.append(emb)
        self.embeddings = new_embeddings

    def search(self, query, n, most_similar, num_threads=None):
        similarities = []
        for i, emb in enumerate(self.embeddings):
            if emb is not None:
                sim = self.cosine_similarity(query, emb)
                similarities.append((i, sim))
        if most_similar:
            similarities.sort(key=lambda x: x[1], reverse=True)
        else:
            similarities.sort(key=lambda x: x[1])
        results = similarities[:n]
        return [{'id': r[0], 'score': r[1]} for r in results]

    def cosine_similarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def load_embeddings(paths):
    embeddings = {}
    for path in paths:
        idx = int(os.path.basename(path).split('-')[1].split('.')[0])
        with open(path, 'rb') as f:
            data = f.read()
            embeddings[idx] = np.array(struct.unpack('<1024f', data), dtype=np.float32)
    return embeddings

def main():
    print("=== Python Embedlet Equivalent Benchmark ===\n")

    store = EmbedletStore(DIMS)

    # Load template embeddings 0-149
    embedding_files = ["embedding-{0:03d}.dat".format(i) for i in range(150)]
    embeddings_loaded = load_embeddings(embedding_files)

    print("Loading and caching template embeddings...")
    embeddings = [None] * 150
    for i in range(150):
        embeddings[i] = embeddings_loaded[i]

    # Fill store with 100,000 embeddings by cycling
    total_embeddings = 100000
    print(f"Filling store with {total_embeddings} embeddings (~409MB)\n")

    import time
    fill_start = time.time()
    for i in range(total_embeddings):
        emb = embeddings[i % 150]
        idx = store.append(emb, reuse=False)
    fill_time = time.time() - fill_start

    print(f"Time to fill {total_embeddings} embeddings: {fill_time:.2f} seconds")

    # Load query
    query_emb = embeddings[0]
    print("\nLoaded query from embedding-000.dat")
    print("Timing top-5 most similar search...")

    search_start = time.time()
    results = store.search(query_emb, 5, most_similar=True, num_threads=1)
    search_time = time.time() - search_start

    print(f"Search time: {search_time:.2f} seconds")

    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()
