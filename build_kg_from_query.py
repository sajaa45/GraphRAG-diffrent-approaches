#!/usr/bin/env python3
"""
Build Neo4j Knowledge Graph from Vector Store Query Results
Uses LLM (Ollama/Llama) for accurate entity extraction
Supports multiple query types: CEO, revenue, risk factors
"""

import os
import json
import argparse
import requests
from typing import List, Dict, Tuple, Any
from neo4j import GraphDatabase
from query_vector_store import VectorStoreQuery

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_QUERY = "CEO"
TOP_N_CHUNKS = 3
SIMILARITY_THRESHOLD = 0.45
OLLAMA_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "llama3.2:1b"
# ============================================================================


class KnowledgeGraphBuilder:
    """Build Neo4j knowledge graph from query results using LLM"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 ollama_url: str = OLLAMA_URL, ollama_model: str = OLLAMA_MODEL):
        """Initialize Neo4j and Ollama connections"""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        print(f"Connected to Neo4j at {neo4j_uri}")
        print(f"Using Ollama at {ollama_url} with model {ollama_model}")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")
    
    def is_current_position(self, text: str, person_name: str) -> bool:
        """
        Check if the text indicates a current position for the person
        Returns False if text contains past tense indicators
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        person_lower = person_name.lower()
        
        # Find the person's context in the text
        person_idx = text_lower.find(person_lower.split()[0].lower())
        if person_idx == -1:
            return False
        
        # Get surrounding context (200 chars before and after)
        context_start = max(0, person_idx - 200)
        context_end = min(len(text), person_idx + 200)
        context = text_lower[context_start:context_end]
        
        # Past tense indicators - strong signals this is NOT current
        past_indicators = [
            'previously', 'formerly', 'has served', 'had served',
            'served as', 'was the', 'was a', 'from 20', 'to 20',
            'until 20', 'prior to', 'before serving', 'retired'
        ]
        
        # Current position indicators
        current_indicators = [
            'currently', 'serves as', 'is the', 'is a',
            'president and ceo', 'chief executive officer'
        ]
        
        # Check for past indicators
        for indicator in past_indicators:
            if indicator in context:
                return False
        
        # Check for current indicators
        for indicator in current_indicators:
            if indicator in context:
                return True
        
        # Default to False if unclear
        return False
    
    def extract_entities_with_llm(self, text: str, query_type: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM based on query type
        
        Returns:
            List of entity dictionaries with type-specific fields
        """
        if query_type.lower() in ['ceo', 'cfo', 'executive', 'board']:
            return self._extract_people_entities(text, query_type)
        elif 'revenue' in query_type.lower():
            return self._extract_revenue_entities(text)
        elif 'risk' in query_type.lower():
            return self._extract_risk_entities(text)
        else:
            return self._extract_people_entities(text, query_type)
    
    def _extract_people_entities(self, text: str, query_type: str) -> List[Dict[str, Any]]:
        """Extract people/executive entities"""
        prompt = f"""Extract CURRENT {query_type} information from this text. Ignore past/former positions.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"person": "Full Name", "role": "CEO", "organization": "Company Name", "is_current": true}}
]

CRITICAL Rules:
- Extract ONLY people who CURRENTLY hold positions (ignore "Previously", "has served", "from X to Y")
- Extract ONLY actual person names with first AND last name
- role can be: CEO, CFO, Board Member, Director, Executive
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as organization
- Set is_current to true ONLY if the text indicates they CURRENTLY hold the position
- Look for keywords: "Currently", "serves as", "is", present tense verbs
- IGNORE keywords: "Previously", "has served", "from...to", "was", past tense verbs
- Return empty array [] if no CURRENT positions found
"""
        
        entities_data = self._call_llm(prompt)
        if not entities_data:
            return []
        
        # Convert to standardized format
        entities = []
        for entity in entities_data:
            person = entity.get('person', '').strip()
            role = entity.get('role', '').strip()
            org = entity.get('organization', 'Saudi Aramco').strip()
            is_current = entity.get('is_current', False)
            
            # Double-check with text analysis
            if person and len(person) > 2:
                if not self.is_current_position(text, person):
                    continue
                
                if is_current:
                    # Map role to relationship type
                    if 'ceo' in role.lower() or 'chief executive' in role.lower():
                        rel_type = 'CEO_OF'
                    elif 'cfo' in role.lower() or 'chief financial' in role.lower():
                        rel_type = 'CFO_OF'
                    elif 'board' in role.lower() or 'director' in role.lower():
                        rel_type = 'BOARD_MEMBER_OF'
                    else:
                        rel_type = 'WORKS_AT'
                    
                    entities.append({
                        'type': 'person',
                        'person': person,
                        'relationship': rel_type,
                        'organization': org
                    })
        
        return entities
    
    def _extract_revenue_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract revenue/financial entities"""
        prompt = f"""Extract revenue and financial information from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"metric": "Revenue", "value": "123.4", "unit": "billion", "currency": "USD", "year": "2024", "organization": "Saudi Aramco"}}
]

Rules:
- Extract financial metrics like: Revenue, Net Income, Profit, Sales, EBITDA
- Include the numeric value, unit (million/billion), currency, and year
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as organization
- Return empty array [] if no financial data found
"""
        
        entities_data = self._call_llm(prompt)
        if not entities_data:
            return []
        
        entities = []
        for entity in entities_data:
            metric = entity.get('metric', '').strip()
            value = entity.get('value', '').strip()
            unit = entity.get('unit', '').strip()
            currency = entity.get('currency', 'USD').strip()
            year = entity.get('year', '').strip()
            org = entity.get('organization', 'Saudi Aramco').strip()
            
            if metric and value:
                entities.append({
                    'type': 'financial',
                    'metric': metric,
                    'value': value,
                    'unit': unit,
                    'currency': currency,
                    'year': year,
                    'organization': org
                })
        
        return entities
    
    def _extract_risk_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract risk factor entities"""
        prompt = f"""Extract risk factors from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"risk_type": "Market Risk", "description": "brief description", "severity": "High", "organization": "Saudi Aramco"}}
]

Rules:
- Extract risk types like: Market Risk, Operational Risk, Regulatory Risk, Financial Risk, etc.
- Provide a brief description (max 100 chars)
- severity can be: High, Medium, Low, or Unknown
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as organization
- Return empty array [] if no risks found
"""
        
        entities_data = self._call_llm(prompt)
        if not entities_data:
            return []
        
        entities = []
        for entity in entities_data:
            risk_type = entity.get('risk_type', '').strip()
            description = entity.get('description', '').strip()
            severity = entity.get('severity', 'Unknown').strip()
            org = entity.get('organization', 'Saudi Aramco').strip()
            
            if risk_type:
                entities.append({
                    'type': 'risk',
                    'risk_type': risk_type,
                    'description': description,
                    'severity': severity,
                    'organization': org
                })
        
        return entities
    
    def _call_llm(self, prompt: str) -> List[Dict]:
        """Call Ollama LLM and parse JSON response"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1  # Low temperature for consistent extraction
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"    ✗ Ollama error: {response.status_code}")
                return []
            
            result = response.json()
            llm_output = result.get('response', '').strip()
            
            # Extract JSON from response
            json_start = llm_output.find('[')
            json_end = llm_output.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                print(f"    ✗ No JSON found in LLM response")
                return []
            
            json_str = llm_output[json_start:json_end]
            entities_data = json.loads(json_str)
            
            return entities_data
            
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Ollama connection error: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"    ✗ JSON parse error: {e}")
            print(f"    LLM output: {llm_output[:200]}")
            return []
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            return []
    
    def create_person_node(self, session, name: str, source_chunk: str, similarity: float):
        """Create or update a Person node"""
        query = """
        MERGE (p:Person {name: $name})
        ON CREATE SET 
            p.created_at = datetime(),
            p.source_chunks = [$source_chunk],
            p.max_similarity = $similarity
        ON MATCH SET
            p.source_chunks = CASE 
                WHEN NOT $source_chunk IN p.source_chunks 
                THEN p.source_chunks + $source_chunk 
                ELSE p.source_chunks 
            END,
            p.max_similarity = CASE 
                WHEN $similarity > p.max_similarity THEN $similarity 
                ELSE p.max_similarity 
            END
        RETURN p
        """
        result = session.run(query, name=name, source_chunk=source_chunk, similarity=similarity)
        return result.single()
    
    def create_organization_node(self, session, name: str):
        """Create or update an Organization node"""
        query = """
        MERGE (o:Organization {name: $name})
        ON CREATE SET o.created_at = datetime()
        RETURN o
        """
        result = session.run(query, name=name)
        return result.single()
    
    def create_relationship(self, session, person_name: str, org_name: str, 
                          rel_type: str, source_chunk: str, similarity: float):
        """Create relationship between person and organization"""
        query = f"""
        MATCH (p:Person {{name: $person_name}})
        MATCH (o:Organization {{name: $org_name}})
        MERGE (p)-[r:{rel_type}]->(o)
        ON CREATE SET 
            r.created_at = datetime(),
            r.source_chunks = [$source_chunk],
            r.confidence = $similarity
        ON MATCH SET
            r.source_chunks = CASE 
                WHEN NOT $source_chunk IN r.source_chunks 
                THEN r.source_chunks + $source_chunk 
                ELSE r.source_chunks 
            END,
            r.confidence = CASE 
                WHEN $similarity > r.confidence THEN $similarity 
                ELSE r.confidence 
            END
        RETURN r
        """
        result = session.run(query, 
                           person_name=person_name, 
                           org_name=org_name,
                           source_chunk=source_chunk,
                           similarity=similarity)
        return result.single()
    
    def build_graph_from_chunks(self, chunks: List[Dict], query_type: str):
        """Build knowledge graph from query result chunks using LLM"""
        print(f"\nBuilding knowledge graph from {len(chunks)} chunks using LLM...")
        print(f"Query type: {query_type}")
        print("="*60)
        
        total_entities = 0
        total_relationships = 0
        
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks, 1):
                text = chunk['text']
                similarity = chunk['similarity']
                section = chunk['section_title']
                
                print(f"\nChunk {i} (similarity: {similarity:.3f}):")
                print(f"  Section: {section}")
                print(f"  Extracting entities with LLM...")
                
                # Extract entities using LLM
                entities = self.extract_entities_with_llm(text, query_type)
                
                if entities:
                    print(f"  ✓ Found {len(entities)} entities:")
                    
                    for entity in entities:
                        if entity['type'] == 'person':
                            person_name = entity['person']
                            rel_type = entity['relationship']
                            org_name = entity['organization']
                            
                            print(f"    - {person_name} --[{rel_type}]--> {org_name}")
                            
                            # Create nodes
                            self.create_person_node(session, person_name, text[:200], similarity)
                            self.create_organization_node(session, org_name)
                            
                            # Create relationship
                            self.create_relationship(session, person_name, org_name, 
                                                   rel_type, text[:200], similarity)
                            
                            total_entities += 1
                            total_relationships += 1
                else:
                    print(f"    ✗ No entities extracted")
        
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH BUILT")
        print("="*60)
        print(f"Total entities extracted: {total_entities}")
        print(f"Total relationships created: {total_relationships}")
        
        # Show graph stats
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            print("\nNode counts:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            print("\nRelationship counts:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Neo4j KG from vector store query using LLM"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Search query (default: '{DEFAULT_QUERY}')"
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial_docs",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--db-path", "-d",
        default="./chroma_db",
        help="ChromaDB path"
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "Lexical12345"),
        help="Neo4j password"
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", OLLAMA_URL),
        help=f"Ollama URL (default: {OLLAMA_URL})"
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
        help=f"Ollama model (default: {OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_CHUNKS,
        help=f"Number of top chunks to use (default: {TOP_N_CHUNKS})"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing graph before building"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("KNOWLEDGE GRAPH BUILDER FROM VECTOR STORE (LLM-based)")
    print("="*60)
    
    # Step 1: Query vector store
    print(f"\nStep 1: Querying vector store for '{args.query}'...")
    query_interface = VectorStoreQuery(
        collection_name=args.collection,
        persist_directory=args.db_path
    )
    
    results = query_interface.hierarchical_query(
        args.query,
        n_sections=2,
        n_chunks_per_section=args.top_n
    )
    
    if not results['chunks']:
        print("✗ No chunks found for query")
        return
    
    # Filter top N chunks with similarity threshold
    top_chunks = [
        c for c in results['chunks']
        if c['similarity'] >= SIMILARITY_THRESHOLD
    ][:args.top_n]
    
    print(f"✓ Found {len(results['chunks'])} chunks, using top {len(top_chunks)}")
    
    # Step 2: Build knowledge graph using LLM
    print(f"\nStep 2: Building Neo4j knowledge graph with LLM extraction...")
    kg_builder = KnowledgeGraphBuilder(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.ollama_url,
        args.ollama_model
    )
    
    try:
        if args.clear:
            kg_builder.clear_database()
        
        kg_builder.build_graph_from_chunks(top_chunks, args.query)
        
        print("\n✓ Knowledge graph built successfully!")
        print(f"\nView graph at: http://localhost:7474")
        print(f"  Username: {args.neo4j_user}")
        print(f"  Password: {args.neo4j_password}")
        
    finally:
        kg_builder.close()


if __name__ == "__main__":
    main()
