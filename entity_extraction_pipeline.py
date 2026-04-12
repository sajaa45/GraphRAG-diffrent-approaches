#!/usr/bin/env python3
"""
Entity and Relationship Extraction Pipeline
Uses vector store for retrieval, then extracts entities/relationships for Neo4j
"""

import json
import re
from typing import List, Dict, Any, Tuple
from query_vector_store import VectorStoreQuery
from neo4j import GraphDatabase
import requests
import time


class EntityExtractor:
    """Extract entities and relationships from text using LLM"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.ollama_url = ollama_url
        self.model = model
        
        # Define entity types and relationship types for financial documents
        self.entity_types = [
            "COMPANY", "PERSON", "ORGANIZATION", "LOCATION",
            "FINANCIAL_METRIC", "DATE", "AMOUNT", "PERCENTAGE",
            "PRODUCT", "SERVICE", "INDUSTRY", "REGULATION"
        ]
        
        self.relationship_types = [
            "REPORTED", "INCREASED", "DECREASED", "OWNS", "OPERATES",
            "LOCATED_IN", "PART_OF", "ACQUIRED", "INVESTED_IN",
            "REGULATED_BY", "COMPETES_WITH", "SUPPLIES_TO"
        ]
    
    def extract_entities_and_relationships(self, text: str, chunk_metadata: Dict) -> Dict:
        """
        Extract entities and relationships from text using LLM
        
        Args:
            text: Text to extract from
            chunk_metadata: Metadata about the chunk (page, section, etc.)
            
        Returns:
            Dictionary with entities and relationships
        """
        prompt = f"""Extract entities and relationships from the following financial document text.

ENTITY TYPES: {', '.join(self.entity_types)}
RELATIONSHIP TYPES: {', '.join(self.relationship_types)}

TEXT:
{text[:2000]}  # Limit to avoid token limits

Extract in JSON format:
{{
  "entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE", "properties": {{"key": "value"}}}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "type": "RELATIONSHIP_TYPE", "properties": {{"key": "value"}}}}
  ]
}}

Focus on:
- Financial metrics (revenue, profit, assets, etc.)
- Companies and organizations
- Key people and their roles
- Locations and operations
- Dates and time periods
- Amounts and percentages

JSON OUTPUT:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1000}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result_text = response.json().get("response", "")
                
                # Try to parse JSON from response
                try:
                    # Extract JSON from response (might have extra text)
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        extracted = json.loads(json_match.group())
                        
                        # Add metadata to entities
                        for entity in extracted.get('entities', []):
                            entity['source_chunk'] = chunk_metadata.get('chunk_id')
                            entity['source_page'] = chunk_metadata.get('page')
                            entity['source_section'] = chunk_metadata.get('section_path')
                        
                        # Add metadata to relationships
                        for rel in extracted.get('relationships', []):
                            rel['source_chunk'] = chunk_metadata.get('chunk_id')
                            rel['source_page'] = chunk_metadata.get('page')
                        
                        return extracted
                except json.JSONDecodeError:
                    print(f"  Warning: Could not parse JSON from LLM response")
                    return {"entities": [], "relationships": []}
            
            return {"entities": [], "relationships": []}
            
        except Exception as e:
            print(f"  Error extracting entities: {e}")
            return {"entities": [], "relationships": []}
    
    def extract_simple_entities(self, text: str, chunk_metadata: Dict) -> Dict:
        """
        Fallback: Simple rule-based entity extraction
        Faster but less accurate than LLM
        """
        entities = []
        relationships = []
        
        # Extract amounts (SAR, USD, etc.)
        amount_pattern = r'(SAR|USD|€|£)\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?'
        for match in re.finditer(amount_pattern, text, re.IGNORECASE):
            entities.append({
                "name": match.group(),
                "type": "AMOUNT",
                "properties": {"value": match.group()},
                "source_chunk": chunk_metadata.get('chunk_id'),
                "source_page": chunk_metadata.get('page')
            })
        
        # Extract percentages
        percentage_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(percentage_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "PERCENTAGE",
                "properties": {"value": match.group()},
                "source_chunk": chunk_metadata.get('chunk_id'),
                "source_page": chunk_metadata.get('page')
            })
        
        # Extract dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "DATE",
                "properties": {"value": match.group()},
                "source_chunk": chunk_metadata.get('chunk_id'),
                "source_page": chunk_metadata.get('page')
            })
        
        # Extract company names (capitalized words)
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Company|Corporation|Inc\.|Ltd\.|LLC|Group)\b'
        for match in re.finditer(company_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "COMPANY",
                "properties": {"name": match.group()},
                "source_chunk": chunk_metadata.get('chunk_id'),
                "source_page": chunk_metadata.get('page')
            })
        
        return {"entities": entities, "relationships": relationships}


class KnowledgeGraphBuilder:
    """Build Neo4j knowledge graph from extracted entities"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.create_constraints()
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def create_constraints(self):
        """Create constraints for entity nodes"""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_ref IF NOT EXISTS FOR (c:ChunkReference) REQUIRE c.chunk_id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # Constraint may already exist
    
    def add_entity(self, entity: Dict):
        """Add entity node to graph"""
        with self.driver.session() as session:
            # Create unique ID for entity
            entity_id = f"{entity['type']}_{entity['name'].replace(' ', '_')}"
            
            session.run("""
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.type = $type,
                    e.source_chunk = $source_chunk,
                    e.source_page = $source_page,
                    e.source_section = $source_section,
                    e.properties = $properties,
                    e.updated_at = datetime()
                WITH e
                MERGE (c:ChunkReference {chunk_id: $source_chunk})
                MERGE (e)-[:EXTRACTED_FROM]->(c)
            """,
                id=entity_id,
                name=entity['name'],
                type=entity['type'],
                source_chunk=entity.get('source_chunk'),
                source_page=entity.get('source_page'),
                source_section=entity.get('source_section'),
                properties=json.dumps(entity.get('properties', {}))
            )
    
    def add_relationship(self, relationship: Dict):
        """Add relationship between entities"""
        with self.driver.session() as session:
            source_id = f"{relationship.get('source_type', 'Entity')}_{relationship['source'].replace(' ', '_')}"
            target_id = f"{relationship.get('target_type', 'Entity')}_{relationship['target'].replace(' ', '_')}"
            
            session.run("""
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATED {type: $rel_type}]->(target)
                SET r.properties = $properties,
                    r.source_chunk = $source_chunk,
                    r.updated_at = datetime()
            """,
                source_id=source_id,
                target_id=target_id,
                rel_type=relationship['type'],
                properties=json.dumps(relationship.get('properties', {})),
                source_chunk=relationship.get('source_chunk')
            )
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()['count']
            
            result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            result = session.run("MATCH (c:ChunkReference) RETURN count(c) as count")
            chunk_count = result.single()['count']
            
            return {
                "entities": entity_count,
                "relationships": rel_count,
                "chunks": chunk_count
            }


class VectorStoreEntityPipeline:
    """Complete pipeline: Vector Store → Entity Extraction → Knowledge Graph"""
    
    def __init__(self,
                 vector_collection: str = "financial_docs",
                 vector_db_path: str = "./chroma_db",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_username: str = "neo4j",
                 neo4j_password: str = "password",
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2:1b",
                 use_llm: bool = False):
        """
        Initialize pipeline
        
        Args:
            vector_collection: ChromaDB collection name
            vector_db_path: Path to ChromaDB
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            ollama_url: Ollama API URL
            ollama_model: Model for entity extraction
            use_llm: Use LLM for extraction (slower) or rule-based (faster)
        """
        print("Initializing Vector Store Entity Pipeline...")
        
        # Initialize components
        self.query_interface = VectorStoreQuery(
            collection_name=vector_collection,
            persist_directory=vector_db_path
        )
        
        self.extractor = EntityExtractor(ollama_url, ollama_model)
        self.kg_builder = KnowledgeGraphBuilder(neo4j_uri, neo4j_username, neo4j_password)
        self.use_llm = use_llm
        
        print("✓ Pipeline initialized")
    
    def process_query(self, query: str, n_sections: int = 2, n_chunks_per_section: int = 3) -> Dict:
        """
        Process query: retrieve relevant chunks and extract entities
        
        Args:
            query: Search query
            n_sections: Number of sections to retrieve
            n_chunks_per_section: Chunks per section
            
        Returns:
            Dictionary with results
        """
        print(f"\nProcessing query: '{query}'")
        print("="*60)
        
        # Step 1: Retrieve relevant chunks using hierarchical search
        print("\n[1] Retrieving relevant chunks from vector store...")
        results = self.query_interface.hierarchical_query(
            query,
            n_sections=n_sections,
            n_chunks_per_section=n_chunks_per_section
        )
        
        print(f"  Retrieved {len(results['chunks'])} chunks from {len(results['sections'])} sections")
        
        # Step 2: Extract entities from each chunk
        print("\n[2] Extracting entities and relationships...")
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(results['chunks'], 1):
            print(f"  Processing chunk {i}/{len(results['chunks'])} (Page {chunk['page']})...")
            
            # Extract entities
            if self.use_llm:
                extracted = self.extractor.extract_entities_and_relationships(
                    chunk['text'],
                    chunk
                )
            else:
                extracted = self.extractor.extract_simple_entities(
                    chunk['text'],
                    chunk
                )
            
            all_entities.extend(extracted.get('entities', []))
            all_relationships.extend(extracted.get('relationships', []))
            
            print(f"    Found {len(extracted.get('entities', []))} entities, {len(extracted.get('relationships', []))} relationships")
        
        print(f"\n  Total: {len(all_entities)} entities, {len(all_relationships)} relationships")
        
        # Step 3: Add to knowledge graph
        print("\n[3] Building knowledge graph...")
        
        for entity in all_entities:
            self.kg_builder.add_entity(entity)
        
        for relationship in all_relationships:
            self.kg_builder.add_relationship(relationship)
        
        stats = self.kg_builder.get_statistics()
        print(f"  Graph now has: {stats['entities']} entities, {stats['relationships']} relationships")
        
        return {
            "query": query,
            "chunks_retrieved": len(results['chunks']),
            "sections_retrieved": len(results['sections']),
            "entities_extracted": len(all_entities),
            "relationships_extracted": len(all_relationships),
            "graph_stats": stats,
            "chunks": results['chunks'],
            "entities": all_entities,
            "relationships": all_relationships
        }
    
    def batch_process(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries"""
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(queries)}")
            print(f"{'='*80}")
            
            result = self.process_query(query)
            results.append(result)
        
        return results
    
    def close(self):
        """Close connections"""
        self.kg_builder.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract entities from vector store and build knowledge graph"
    )
    parser.add_argument(
        "--query", "-q",
        help="Query to process"
    )
    parser.add_argument(
        "--queries-file",
        help="File with queries (one per line)"
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial_docs",
        help="Vector store collection"
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Vector database path"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for entity extraction (slower but better)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VectorStoreEntityPipeline(
        vector_collection=args.collection,
        vector_db_path=args.db_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        use_llm=args.use_llm
    )
    
    try:
        # Process queries
        if args.query:
            results = [pipeline.process_query(args.query)]
        elif args.queries_file:
            with open(args.queries_file) as f:
                queries = [line.strip() for line in f if line.strip()]
            results = pipeline.batch_process(queries)
        else:
            # Default test queries
            queries = [
                "What was the company's revenue?",
                "Who are the key executives?",
                "What are the main risk factors?"
            ]
            results = pipeline.batch_process(queries)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        total_entities = sum(r['entities_extracted'] for r in results)
        total_relationships = sum(r['relationships_extracted'] for r in results)
        
        print(f"Queries processed: {len(results)}")
        print(f"Total entities extracted: {total_entities}")
        print(f"Total relationships extracted: {total_relationships}")
        
        if results:
            final_stats = results[-1]['graph_stats']
            print(f"\nFinal graph statistics:")
            print(f"  Entities: {final_stats['entities']}")
            print(f"  Relationships: {final_stats['relationships']}")
            print(f"  Source chunks: {final_stats['chunks']}")
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
