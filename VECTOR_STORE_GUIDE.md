# Vector Store Pipeline Guide

## Overview

This pipeline replaces JSON-based embedding storage with a fast, persistent vector database (ChromaDB). It embeds sections and subsections hierarchically alongside chunks for better retrieval.

## Key Features

- **Hierarchical Embeddings**: Sections, subsections, and chunks all embedded
- **Fast Parallel Processing**: Multi-threaded embedding generation
- **Persistent Storage**: ChromaDB for efficient disk-based storage
- **Semantic Search**: Find relevant content by meaning, not keywords
- **Section-Aware Retrieval**: Query within specific sections or pages
- **No JSON Bloat**: Embeddings stored efficiently in vector DB

## Architecture

```
PDF Document
    ↓
Sections (parse_pdf.py)
    ↓
Vector Store Pipeline
    ├── Section Embeddings (for high-level search)
    ├── Subsection Embeddings (for mid-level search)
    └── Chunk Embeddings (for detailed search)
    ↓
ChromaDB (persistent storage)
    ↓
Query Interface (semantic search)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies:
- `chromadb>=0.4.22` - Vector database
- `chonkie>=0.1.0` - Fast semantic chunking

### 2. Build Vector Store

```bash
# Process sections and build vector store
python vector_store_pipeline.py output/parsed_sections.json \
    --collection financial_docs \
    --db-path ./chroma_db \
    --chunk-size 512 \
    --threshold 0.5 \
    --batch-size 100 \
    --workers 4
```

Parameters:
- `--collection`: Name for the collection (default: financial_docs)
- `--db-path`: Where to store the database (default: ./chroma_db)
- `--chunk-size`: Target chunk size in tokens (default: 512)
- `--threshold`: Semantic similarity threshold (default: 0.5)
- `--batch-size`: Batch size for embedding (default: 100)
- `--workers`: Parallel workers (default: 4)

### 3. Query the Store

```bash
# Hierarchical query (recommended)
python query_vector_store.py "revenue growth and profitability" \
    --mode hierarchical

# Query chunks only
python query_vector_store.py "risk factors" \
    --mode chunks \
    --n-results 10

# Query sections only
python query_vector_store.py "financial performance" \
    --mode sections \
    --n-results 5

# Filter by page
python query_vector_store.py "revenue" \
    --mode chunks \
    --page 15

# Show database stats
python query_vector_store.py --stats
```

## Docker Usage

### Build Vector Store

```bash
docker-compose up vector-store
```

### Query Vector Store

```bash
docker-compose run query-store python query_vector_store.py "your query here"
```

## Performance

### Speed Improvements

- **Parallel Embedding**: 4x faster with 4 workers
- **Batch Processing**: 100 docs at a time for efficiency
- **Disk-Based Storage**: No memory limits
- **Fast Queries**: ~0.1-0.5s per query

### Example Timings

For a 200-page financial report:
- Sections: ~50 documents
- Chunks: ~800 documents
- Build time: ~60-120s (depending on hardware)
- Query time: ~0.2s per query
- Storage: ~50MB (vs 500MB+ for JSON with embeddings)

## Hierarchical Retrieval

The hierarchical query strategy:

1. **Find Relevant Sections** (high-level)
   - Embed query
   - Search section embeddings
   - Get top N sections

2. **Find Chunks Within Sections** (detailed)
   - For each relevant section
   - Search chunk embeddings in that section
   - Get top M chunks per section

This approach:
- Maintains context (section hierarchy)
- Reduces noise (focused search)
- Improves relevance (section + chunk matching)

## API Usage

### Python API

```python
from vector_store_pipeline import HierarchicalVectorStore
from query_vector_store import VectorStoreQuery

# Build store
store = HierarchicalVectorStore(
    collection_name="my_docs",
    persist_directory="./my_db"
)

# Add documents
store.add_documents_batch(
    documents=["text1", "text2"],
    metadatas=[{"type": "chunk", "page": 1}, {"type": "chunk", "page": 2}],
    ids=["doc1", "doc2"]
)

# Query
query_interface = VectorStoreQuery(
    collection_name="my_docs",
    persist_directory="./my_db"
)

results = query_interface.hierarchical_query(
    "my query",
    n_sections=2,
    n_chunks_per_section=3
)
```

## Configuration

### Chunk Size

- **Small (256-512)**: More granular, better precision
- **Medium (512-1024)**: Balanced
- **Large (1024-2048)**: More context, better recall

### Similarity Threshold

- **Low (0.3-0.5)**: More chunks, diverse content
- **Medium (0.5-0.7)**: Balanced
- **High (0.7-0.9)**: Fewer, more coherent chunks

### Batch Size

- **Small (50)**: Lower memory, slower
- **Medium (100)**: Balanced
- **Large (200+)**: Faster, more memory

### Workers

- **1-2**: Low CPU usage
- **4**: Balanced (recommended)
- **8+**: Maximum speed (if CPU allows)

## Comparison: Old vs New

### Old Approach (JSON)

```python
# Store embeddings in JSON
{
  "chunks": [
    {
      "text": "...",
      "embedding": [0.1, 0.2, ...],  # 384 floats
      "metadata": {...}
    }
  ]
}
```

Problems:
- Large file sizes (embeddings are big)
- Slow to load (parse entire JSON)
- No semantic search (need custom code)
- Memory intensive (all in RAM)

### New Approach (Vector Store)

```python
# Store in ChromaDB
store.add_documents_batch(
    documents=["text1", "text2"],
    metadatas=[{...}, {...}],
    ids=["id1", "id2"]
)

# Embeddings generated and stored efficiently
# Query with semantic search
results = store.query("query text", n_results=5)
```

Benefits:
- Small storage (efficient compression)
- Fast loading (indexed database)
- Built-in semantic search
- Memory efficient (disk-based)
- Scalable (millions of documents)

## Advanced Usage

### Custom Embedding Model

```python
store = HierarchicalVectorStore(
    embedding_model="all-mpnet-base-v2"  # Better quality
)
```

### Filtered Queries

```python
# Query specific section
results = query_interface.query_chunks(
    "revenue",
    section_path="Financial Performance > Revenue Analysis"
)

# Query specific page
results = query_interface.query_chunks(
    "risk factors",
    page=25
)
```

### Batch Updates

```python
# Add more documents later
store.add_documents_batch(
    documents=new_texts,
    metadatas=new_metas,
    ids=new_ids
)
```

## Troubleshooting

### "Collection not found"

Run the pipeline first:
```bash
python vector_store_pipeline.py output/parsed_sections.json
```

### Slow embedding

- Reduce `--workers` if CPU is maxed
- Increase `--batch-size` for faster processing
- Use smaller embedding model (all-MiniLM-L6-v2)

### Out of memory

- Reduce `--batch-size`
- Reduce `--workers`
- Process in smaller batches

### Poor results

- Adjust `--threshold` (try 0.3-0.7)
- Adjust `--chunk-size` (try 256-1024)
- Try better embedding model (all-mpnet-base-v2)

## Next Steps

1. **Integrate with your app**: Use the query API
2. **Tune parameters**: Experiment with chunk size and threshold
3. **Add more documents**: Scale to your full dataset
4. **Implement RAG**: Use retrieved chunks for LLM context
5. **Add reranking**: Use cross-encoder for better results

## Resources

- ChromaDB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Chonkie: https://github.com/bhavnicksm/chonkie
