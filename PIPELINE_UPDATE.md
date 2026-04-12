# Pipeline Update - Unified Chunking System

## Changes Made

### 1. Refactored `unified_pipeline.py` to use `chunking.py`

The unified pipeline now calls the production chunking system from `chunking.py` instead of duplicating the LlamaIndex setup.

**Benefits:**
- Single source of truth for chunking logic
- Automatic filtering of low-quality chunks:
  - Small chunks (< 30 characters)
  - Decorative content (bullets, symbols)
  - Table of contents chunks
  - Meaningless content
  - Repetitive chunks
- Consistent chunking behavior across the system
- Easier to maintain and update

### 2. Fixed `query_vector_store.py` hierarchical query

Updated the `hierarchical_query` method to work with chunk-only storage:
- Fetches multiple chunks via similarity search
- Groups chunks by `section_title` metadata
- Ranks sections by best chunk similarity
- Returns top N sections with M best chunks each

**No longer requires separate section embeddings!**

### 3. Added Docker Compose query service

Updated `docker-compose.yml` to support custom queries:
```bash
# Default query
docker-compose up --build query-store

# Custom query
QUERY="financial performance" docker-compose up --build query-store
```

## How to Use

### Step 1: Parse PDF (if not done)
```bash
docker-compose up parse-pdf
```

### Step 2: Build Vector Store
```bash
docker-compose up --build vector-store
```

This will:
- Load sections from `output/parsed_sections.json`
- Load page text from source JSON
- Chunk each section using LlamaIndex with filtering
- Store embeddings directly in ChromaDB at `./chroma_db/`
- Save lightweight metadata (no embeddings in JSON)

### Step 3: Query Vector Store
```bash
# Rebuild to get latest code
docker-compose up --build query-store

# Or with custom query
QUERY="revenue growth" docker-compose up --build query-store
```

## What Gets Filtered

The chunking system automatically filters out:

1. **Small chunks** (< 30 chars) - likely just titles
2. **Decorative chunks** (> 30% symbols/bullets) - formatting noise
3. **Table of contents** (similarity > 0.7) - navigation elements
4. **Meaningless content** - empty or navigation text
5. **Repetitive chunks** (< 30% unique words) - redundant content

## Architecture

```
parse_pdf.py
    ↓ (creates parsed_sections.json)
unified_pipeline.py
    ↓ (uses chunking.py)
chunking.py (llamaindex_chunker)
    ↓ (filters + embeds)
ChromaDB (vector store)
    ↓ (queries)
query_vector_store.py
```

## Files Modified

1. `unified_pipeline.py` - Now imports from `chunking.py`
2. `query_vector_store.py` - Fixed hierarchical query for chunk-only storage
3. `docker-compose.yml` - Added query parameter support
4. `VECTOR_QUERY_GUIDE.md` - New usage guide

## Next Steps

To see the vector store in action:

1. Rebuild containers: `docker-compose build`
2. Run vector store: `docker-compose up vector-store`
3. Query it: `docker-compose up query-store`

The query will now properly discover sections from chunks and show relevant results!
