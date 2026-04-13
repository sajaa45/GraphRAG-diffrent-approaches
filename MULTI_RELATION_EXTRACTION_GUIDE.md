# Multi-Relation Knowledge Graph Extraction Guide

## Overview

This framework generalizes the CEO extraction pipeline into a flexible multi-relation extraction system. It uses the same embedding-based hierarchical retrieval strategy (section → chunk) but supports multiple relation types through configuration.

## Architecture

### Core Components

1. **relation_extraction_config.py** - Configuration for all relation types
   - Defines section/chunk keywords for each relation
   - Provides LLM prompt templates
   - Contains entity parsers for standardization

2. **multi_relation_kg_builder.py** - Main extraction pipeline
   - Hierarchical retrieval (section → chunk filtering)
   - LLM-based entity extraction
   - Dynamic Neo4j graph creation

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RELATION CONFIGURATION                                   │
│    - Section keywords (e.g., "board governance directors")  │
│    - Chunk keywords (e.g., "CEO chief executive")           │
│    - LLM extraction prompt                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. HIERARCHICAL RETRIEVAL                                   │
│    Step 1: Embed section keywords → Find relevant sections  │
│    Step 2: Embed chunk keywords → Find chunks in sections   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM ENTITY EXTRACTION                                    │
│    - Use relation-specific prompt on each chunk             │
│    - Parse LLM output to standardized entity format         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. NEO4J GRAPH CREATION                                     │
│    - Create source/target nodes dynamically                 │
│    - Create relationships with metadata                     │
└─────────────────────────────────────────────────────────────┘
```

## Supported Relations

### 1. CEO (Person → Organization)
- **Section keywords**: "overview board governance directors leadership management executive"
- **Chunk keywords**: "CEO chief executive officer"
- **Extracts**: Current CEO with role and organization

### 2. COMPETES_WITH (Company → Company)
- **Section keywords**: "competition competitive landscape business market industry"
- **Chunk keywords**: "compete competitors rivals competitive"
- **Extracts**: Competitor relationships with context

### 3. HAS_METRIC (Company → Metric)
- **Section keywords**: "financial highlights results operations MD&A performance"
- **Chunk keywords**: "revenue EBITDA net income profit growth earnings"
- **Extracts**: Financial metrics with values, units, currency, year

### 4. FACES_RISK (Company → Risk)
- **Section keywords**: "risk factors uncertainties"
- **Chunk keywords**: "risk uncertainty may adversely affect"
- **Extracts**: Risk factors with description and severity

### 5. OPERATES_IN (Company → Industry)
- **Section keywords**: "business overview segments industry operations"
- **Chunk keywords**: "industry sector market segment"
- **Extracts**: Industry and sector information

### 6. OFFERS (Company → Product)
- **Section keywords**: "products services solutions segments offerings"
- **Chunk keywords**: "offer provide product service solution"
- **Extracts**: Products/services with category and description

## Usage

### List Available Relations

```bash
python multi_relation_kg_builder.py --list
```

### Extract Single Relation

```bash
python multi_relation_kg_builder.py CEO
```

### Extract Multiple Relations

```bash
python multi_relation_kg_builder.py CEO COMPETES_WITH HAS_METRIC
```

### Extract All Relations

```bash
python multi_relation_kg_builder.py --all
```

### Clear Graph Before Extraction

```bash
python multi_relation_kg_builder.py --all --clear
```

### Custom Configuration

```bash
python multi_relation_kg_builder.py CEO \
  --collection financial_docs \
  --db-path ./chroma_db \
  --neo4j-uri bolt://localhost:7687 \
  --ollama-url http://localhost:11434 \
  --ollama-model llama3.2:1b
```

## Adding New Relations

To add a new relation type, edit `relation_extraction_config.py`:

### Step 1: Create Entity Parser

```python
def parse_my_entity(entity: Dict) -> Dict:
    """Parse custom entity from LLM output"""
    # Extract fields from LLM response
    field1 = entity.get('field1', '').strip()
    field2 = entity.get('field2', '').strip()
    
    if not field1:
        return None
    
    # Return standardized format
    return {
        'source': {'type': 'SourceType', 'name': field1},
        'target': {
            'type': 'TargetType',
            'name': field2,
            'properties': {'extra': 'data'}
        },
        'relationship': 'MY_RELATIONSHIP',
        'properties': {}
    }
```

### Step 2: Add Relation Configuration

```python
RELATION_CONFIGS = {
    # ... existing configs ...
    
    'MY_RELATION': RelationConfig(
        name='MY_RELATION',
        source_entity_type='SourceType',
        target_entity_type='TargetType',
        relationship_type='MY_RELATIONSHIP',
        section_keywords='keyword1 keyword2 keyword3',
        chunk_keywords='specific chunk keywords',
        extraction_prompt_template="""Extract information from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"field1": "value1", "field2": "value2"}}
]

Rules:
- Rule 1
- Rule 2
- Return empty array [] if nothing found
""",
        entity_parser=parse_my_entity
    )
}
```

### Step 3: Use It

```bash
python multi_relation_kg_builder.py MY_RELATION
```

## Key Features

### 1. No Code Duplication
- Single hierarchical retrieval function for all relations
- Reusable LLM extraction logic
- Dynamic Neo4j operations

### 2. Configuration-Driven
- All relation-specific logic in config file
- Easy to add/modify relations
- No changes to main pipeline code

### 3. Embedding-Based Filtering
- Section-level filtering reduces search space
- Chunk-level semantic search for precision
- Same strategy as original CEO extraction

### 4. Flexible Entity Types
- Supports any source/target entity types
- Dynamic node creation based on config
- Properties stored with nodes and relationships

### 5. Metadata Tracking
- Source chunks stored with relationships
- Confidence scores from similarity
- Timestamps for creation/updates

## Example Output

```
==============================================================
MULTI-RELATION EXTRACTION SUMMARY
==============================================================
CEO: 2 entities, 2 relationships
COMPETES_WITH: 5 entities, 5 relationships
HAS_METRIC: 12 entities, 12 relationships
FACES_RISK: 8 entities, 8 relationships
OPERATES_IN: 3 entities, 3 relationships
OFFERS: 7 entities, 7 relationships

Total: 37 entities, 37 relationships

==============================================================
KNOWLEDGE GRAPH STATISTICS
==============================================================

Nodes:
  Person: 2
  Company: 8
  Metric: 12
  Risk: 8
  Industry: 3
  Product: 7

Relationships:
  CEO_OF: 2
  COMPETES_WITH: 5
  HAS_METRIC: 12
  FACES_RISK: 8
  OPERATES_IN: 3
  OFFERS: 7
```

## Comparison with Original

### Original (build_kg_from_query.py)
- ✗ Hardcoded CEO extraction logic
- ✗ Manual query type checking
- ✗ Separate extraction methods per type
- ✓ Hierarchical retrieval working

### New (multi_relation_kg_builder.py)
- ✓ Configuration-driven relations
- ✓ Single extraction pipeline
- ✓ Easy to add new relations
- ✓ Same hierarchical retrieval
- ✓ Cleaner, more maintainable code

## Performance

- Same embedding-based filtering as original
- Efficient section → chunk narrowing
- Parallel-ready (can process chunks concurrently)
- Reuses embedding model across relations

## Next Steps

1. Add more relation types as needed
2. Tune section/chunk keywords for better retrieval
3. Adjust similarity thresholds per relation
4. Add validation for extracted entities
5. Implement batch processing for large documents
