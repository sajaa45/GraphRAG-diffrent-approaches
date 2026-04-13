# Docker Commands for Multi-Relation Extraction

## Quick Start

### 1. Start Neo4j
```bash
docker-compose up -d neo4j
```

### 2. Run Multi-Relation Extraction

#### Extract specific relations (default: CEO, COMPETES_WITH, HAS_METRIC)
```bash
docker-compose up multi-relation-kg
```

#### Extract all relations
```bash
RELATIONS="--all" docker-compose up multi-relation-kg
```

#### Extract single relation
```bash
RELATIONS="CEO" docker-compose up multi-relation-kg
```

#### Extract custom set of relations
```bash
RELATIONS="CEO HAS_METRIC FACES_RISK" docker-compose up multi-relation-kg
```

#### List available relations
```bash
docker-compose run --rm multi-relation-kg python multi_relation_kg_builder.py --list
```

## Available Relations

- `CEO` - Extract CEO information
- `COMPETES_WITH` - Extract competitor relationships
- `HAS_METRIC` - Extract financial metrics
- `FACES_RISK` - Extract risk factors
- `OPERATES_IN` - Extract industry/sector information
- `OFFERS` - Extract products/services

## Full Pipeline

### Step 1: Parse PDF
```bash
docker-compose up parse-pdf
```

### Step 2: Build Vector Store
```bash
docker-compose up vector-store
```

### Step 3: Extract Relations
```bash
docker-compose up multi-relation-kg
```

## View Results

Neo4j Browser: http://localhost:7474
- Username: `neo4j`
- Password: `Lexical12345`

### Useful Cypher Queries

#### View all nodes and relationships
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

#### Count nodes by type
```cypher
MATCH (n) RETURN labels(n)[0] as type, count(n) as count
```

#### Count relationships by type
```cypher
MATCH ()-[r]->() RETURN type(r) as type, count(r) as count
```

#### Find CEO
```cypher
MATCH (p:Person)-[r:CEO_OF]->(o:Organization) RETURN p, r, o
```

#### Find competitors
```cypher
MATCH (c1:Company)-[r:COMPETES_WITH]->(c2:Company) RETURN c1, r, c2
```

#### Find financial metrics
```cypher
MATCH (c:Company)-[r:HAS_METRIC]->(m:Metric) RETURN c, r, m
```

#### Find risks
```cypher
MATCH (c:Company)-[r:FACES_RISK]->(risk:Risk) RETURN c, r, risk
```

## Troubleshooting

### Check logs
```bash
docker-compose logs multi-relation-kg
```

### Rebuild container
```bash
docker-compose build multi-relation-kg
```

### Clean restart
```bash
docker-compose down
docker-compose up -d neo4j
docker-compose up multi-relation-kg
```

### Clear Neo4j database
The `--clear` flag is already included in the command, but you can also manually clear:
```bash
docker-compose exec neo4j cypher-shell -u neo4j -p Lexical12345 "MATCH (n) DETACH DELETE n"
```

## Environment Variables

You can customize the extraction by setting environment variables:

```bash
# Custom relations
RELATIONS="CEO OFFERS" docker-compose up multi-relation-kg

# Custom Ollama model
OLLAMA_MODEL="llama3.2:3b" docker-compose up multi-relation-kg

# Custom collection
docker-compose run --rm multi-relation-kg \
  python multi_relation_kg_builder.py CEO \
  --collection my_collection \
  --db-path /app/chroma_db
```

## Performance Tips

1. Start with a single relation to test: `RELATIONS="CEO" docker-compose up multi-relation-kg`
2. Use `--all` only when you need comprehensive extraction
3. Monitor Ollama performance - larger models are more accurate but slower
4. Check Neo4j memory if processing large documents

## Next Steps

After extraction, you can:
1. Query the graph using Cypher in Neo4j Browser
2. Export data for analysis
3. Add more relations by editing `relation_extraction_config.py`
4. Tune keywords for better retrieval accuracy
