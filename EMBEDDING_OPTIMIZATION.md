# Embedding Optimization - Using Pre-computed Embeddings

## Overview

The system has been optimized to use pre-computed embeddings stored during the chunking phase, eliminating the need to re-calculate embeddings during query time.

## Changes Made

### 1. Chunking Phase (chunking.py)
- Embeddings are now generated and stored for each chunk during the chunking process
- Each chunk in the JSON output includes an `embedding` field with the pre-computed vector
- Embeddings are stored alongside chunk text, metadata, and other properties

### 2. Knowledge Graph Storage (neo4j_knowledge_graph.py)
- Updated `create_chunk_nodes()` to store embeddings in Neo4j
- Each Chunk node now has an `embedding` property containing the pre-computed vector
- Embeddings are stored as arrays in Neo4j for efficient retrieval

### 3. Query System (graphrag_system.py)
- Updated `find_similar_chunks()` to use pre-computed embeddings from Neo4j
- Only the query text is embedded at query time
- Chunk embeddings are retrieved from Neo4j (not re-calculated)
- Cosine similarity is computed using the pre-computed chunk embeddings

## Performance Benefits

### Before Optimization:
```
Query Time = Embed Query + Embed ALL Chunks + Calculate Similarity
```
- For 100 chunks: ~2-5 seconds per query
- Embedding model loaded and used for every query
- All chunks re-embedded on every query

### After Optimization:
```
Query Time = Embed Query + Calculate Similarity (using stored embeddings)
```
- For 100 chunks: ~0.1-0.3 seconds per query
- Only query needs embedding
- 10-50x faster query performance

## Workflow

### 1. Initial Setup (One-time)
```bash
# Process JSON to sections
python json_text_processor.py

# Generate chunks with embeddings
python chunking.py

# Build knowledge graph with embeddings
python build_knowledge_graph.py
```

### 2. Query Time (Fast)
```python
from graphrag_system import GraphRAGSystem

# Initialize system
graphrag = GraphRAGSystem(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

# Query (only embeds the query, uses pre-computed chunk embeddings)
query = "What is the company's revenue?"
query_embedding = graphrag.embed_query(query)
similar_chunks = graphrag.find_similar_chunks(query_embedding, top_k=5)
```

## Data Structure

### Chunk JSON Format:
```json
{
  "text": "chunk content...",
  "length": 500,
  "embedding": [0.123, -0.456, 0.789, ...],  // 384 dimensions
  "section_id": "section_001",
  "source_pages": [1, 2],
  "page_range": "1-2",
  "chunk_index_in_section": 1,
  "total_chunks_in_section": 5,
  "method": "LlamaIndex"
}
```

### Neo4j Chunk Node:
```cypher
(:Chunk {
  chunk_id: "chunk_0001",
  content: "chunk content...",
  length: 500,
  embedding: [0.123, -0.456, 0.789, ...],  // Pre-computed
  section_id: "section_001",
  page_range: "1-2",
  chunk_index_in_section: 1,
  total_chunks_in_section: 5,
  method: "LlamaIndex"
})
```

## Testing

Run the test script to verify embeddings are working:
```bash
python test_precomputed_embeddings.py
```

This will:
1. Check if embeddings are stored in the JSON file
2. Verify embeddings are in Neo4j
3. Test query performance using pre-computed embeddings
4. Show similarity scores for top results

## Important Notes

1. **Embedding Model Consistency**: The same embedding model must be used for:
   - Generating chunk embeddings (chunking phase)
   - Generating query embeddings (query phase)
   - Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

2. **Re-chunking**: If you re-run chunking with different parameters:
   - New embeddings will be generated automatically
   - Rebuild the knowledge graph to update Neo4j

3. **Storage**: Embeddings add ~1.5KB per chunk (384 floats × 4 bytes)
   - 1000 chunks ≈ 1.5 MB additional storage
   - Worth it for 10-50x query speedup

## Verification

To verify embeddings are stored correctly:

```python
# Check JSON file
import json
with open('output/SemanticSplitterNodeParser_chunks.json', 'r') as f:
    data = json.load(f)
    chunk = data['chunks'][0]
    print(f"Embedding dimensions: {len(chunk['embedding'])}")
    print(f"Has embedding: {len(chunk['embedding']) > 0}")

# Check Neo4j
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN c.embedding as emb LIMIT 1")
    record = result.single()
    print(f"Neo4j embedding dimensions: {len(record['emb'])}")
```

## Troubleshooting

### No embeddings found in Neo4j
- Re-run `build_knowledge_graph.py` to rebuild with embeddings
- Check that JSON file has embeddings first

### Different embedding dimensions
- Ensure same model is used for chunking and querying
- Default is `all-MiniLM-L6-v2` (384 dimensions)

### Slow queries
- Verify embeddings are being loaded from Neo4j (not re-calculated)
- Check console output - should say "Loaded X chunks with pre-computed embeddings"
