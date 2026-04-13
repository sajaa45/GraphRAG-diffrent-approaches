@echo off
REM Quick test script for multi-relation extraction (Windows)

echo ==========================================
echo Multi-Relation Extraction Test
echo ==========================================

REM Check if Neo4j is running
echo.
echo Step 1: Starting Neo4j...
docker-compose up -d neo4j

echo.
echo Waiting for Neo4j to be ready...
timeout /t 15 /nobreak

REM Test 1: List available relations
echo.
echo ==========================================
echo Test 1: List Available Relations
echo ==========================================
docker-compose run --rm multi-relation-kg python multi_relation_kg_builder.py --list

REM Test 2: Extract CEO only
echo.
echo ==========================================
echo Test 2: Extract CEO Relation
echo ==========================================
set RELATIONS=CEO
docker-compose up multi-relation-kg

REM Test 3: Extract multiple relations
echo.
echo ==========================================
echo Test 3: Extract Multiple Relations
echo ==========================================
set RELATIONS=CEO COMPETES_WITH HAS_METRIC
docker-compose up multi-relation-kg

echo.
echo ==========================================
echo Tests Complete!
echo ==========================================
echo.
echo View results at: http://localhost:7474
echo Username: neo4j
echo Password: Lexical12345
echo.
echo Try these Cypher queries:
echo   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
echo   MATCH (n) RETURN labels(n)[0] as type, count(n) as count
echo   MATCH ()-[r]->() RETURN type(r) as type, count(r) as count

pause
