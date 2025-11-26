import json
import struct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"},
)

# Load source sentences from source.json
with open('source.json', 'r') as f:
    data = json.load(f)

# Extract sentences in order of keys (embeddings-000.dat to embeddings-149.dat)
sentences = [data[key] for key in sorted(data.keys())]

# Generate embeddings for all sentences
embeddings = model.encode(sentences)

# Save each embedding to a file as raw little-endian float32 binary
for i in range(len(embeddings)):
    emb = embeddings[i]
    with open(f'embedding-{i:03d}.dat', 'wb') as f:
        f.write(struct.pack('<1024f', *emb))

print(f"Generated {len(embeddings)} embedding files: embedding-000.dat to embedding-{len(embeddings)-1:03d}.dat")
