#!/usr/bin/env python3
"""
Generalized Multi-Relation Knowledge Graph Builder
Supports multiple relation types using embedding-based hierarchical retrieval
"""

import os
import json
import argparse
import requests
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from relation_extraction_config import (
    RelationConfig, 
    get_relation_config, 
    list_available_relations
)

# ============================================================================
# CONFIGURATION
# ============================================================================
TOP_N_SECTIONS = 2
TOP_N_CHUNKS_PER_SECTION = 3
SECTION_SIMILARITY_THRESHOLD = 0.4
CHUNK_SIMILARITY_THRESHOLD = 0.45
OLLAMA_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "llama3.2:1b"
# ============================================================================


class MultiRelationKGBuilder:
    """Build Neo4j knowledge graph with multiple relation types"""
    
    def __init__(self, 
                 neo4j_uri: str, 
                 neo4j_user: str, 
                 neo4j_password: str,
                 collection_name: str = "financial_docs",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_url: str = OLLAMA_URL,
                 ollama_model: str = OLLAMA_MODEL):
        """Initialize connections"""
        # Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"✓ Connected to Neo4j at {neo4j_uri}")
        
        # Ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        print(f"✓ Using Ollama at {ollama_url} with model {ollama_model}")
        
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Loaded embedding model: {embedding_model}")
        
        # ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(name=collection_name)
        print(f"✓ Connected to ChromaDB collection: {collection_name}")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")
    
    # ========================================================================
    # HIERARCHICAL RETRIEVAL (Section → Chunk)
    # ========================================================================
    
    def hierarchical_retrieval(self, 
                              relation_config: RelationConfig,
                              n_sections: int = TOP_N_SECTIONS,
                              n_chunks_per_section: int = TOP_N_CHUNKS_PER_SECTION,
                              section_threshold: float = SECTION_SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Hierarchical retrieval using relation-specific keywords:
        1. Find relevant sections using section_keywords
        2. Search chunks within those sections using chunk_keywords
        3. Fallback to direct chunk search if no sections match
        """
        print(f"\n{'='*60}")
        print(f"Hierarchical Retrieval: {relation_config.name}")
        print(f"{'='*60}")
        
        # Step 1: Find relevant sections using section keywords
        print(f"\nStep 1: Finding sections with keywords: '{relation_config.section_keywords}'")
        section_embedding = self.embedding_model.encode([relation_config.section_keywords])[0]
        
        section_results = self.collection.query(
            query_embeddings=[section_embedding.tolist()],
            n_results=n_sections,
            where={"type": "section"}
        )
        
        use_sections = False
        
        if not section_results['documents'][0]:
            print("  ✗ No sections found")
        else:
            best_section_similarity = 1 - section_results['distances'][0][0]
            print(f"  Best section similarity: {best_section_similarity:.3f}")
            
            if best_section_similarity >= section_threshold:
                use_sections = True
                print(f"  ✓ Found {len(section_results['documents'][0])} relevant sections:")
                for meta in section_results['metadatas'][0]:
                    print(f"    - {meta['title']}")
            else:
                print(f"  ✗ Below threshold ({best_section_similarity:.3f} < {section_threshold})")
        
        all_chunks = []
        chunk_embedding = self.embedding_model.encode([relation_config.chunk_keywords])[0]
        
        if use_sections:
            # Step 2: Search chunks within sections using chunk keywords
            print(f"\nStep 2: Searching chunks with keywords: '{relation_config.chunk_keywords}'")
            
            for i, (doc, meta, distance) in enumerate(zip(
                section_results['documents'][0],
                section_results['metadatas'][0],
                section_results['distances'][0]
            ), 1):
                section_id = meta['section_id']
                section_title = meta['title']
                section_similarity = 1 - distance
                
                print(f"\n  Section {i}: {section_title}")
                print(f"    Similarity: {section_similarity:.3f}")
                
                # Query chunks in this section
                chunk_results = self.collection.query(
                    query_embeddings=[chunk_embedding.tolist()],
                    n_results=n_chunks_per_section,
                    where={"$and": [{"type": "chunk"}, {"section_id": section_id}]}
                )
                
                if chunk_results['documents'][0]:
                    for j, (chunk_doc, chunk_meta, chunk_dist) in enumerate(zip(
                        chunk_results['documents'][0],
                        chunk_results['metadatas'][0],
                        chunk_results['distances'][0]
                    ), 1):
                        chunk_similarity = 1 - chunk_dist
                        print(f"    Chunk {j}: Similarity {chunk_similarity:.3f}")
                        
                        all_chunks.append({
                            "section_title": section_title,
                            "section_id": section_id,
                            "chunk_index": chunk_meta.get('chunk_index', j),
                            "similarity": chunk_similarity,
                            "text": chunk_doc
                        })
        else:
            # Fallback: Direct chunk search
            print(f"\nStep 2: Direct chunk search (no relevant sections found)")
            print(f"  Searching for chunks with keywords: '{relation_config.chunk_keywords}'")
            
            n_chunks = n_sections * n_chunks_per_section
            chunk_results = self.collection.query(
                query_embeddings=[chunk_embedding.tolist()],
                n_results=n_chunks,
                where={"type": "chunk"}
            )
            
            if chunk_results['documents'][0]:
                print(f"  ✓ Found {len(chunk_results['documents'][0])} chunks")
                
                for doc, meta, distance in zip(
                    chunk_results['documents'][0],
                    chunk_results['metadatas'][0],
                    chunk_results['distances'][0]
                ):
                    chunk_similarity = 1 - distance
                    section_title = meta.get('section_title', 'Unknown')
                    
                    all_chunks.append({
                        "section_title": section_title,
                        "section_id": meta.get('section_id'),
                        "chunk_index": meta.get('chunk_index'),
                        "similarity": chunk_similarity,
                        "text": doc
                    })
            else:
                print(f"  ✗ No chunks found")
        
        print(f"\n  ✓ Retrieved {len(all_chunks)} chunks total")
        return all_chunks
    
    # ========================================================================
    # LLM EXTRACTION
    # ========================================================================
    
    def extract_entities_with_llm(self, 
                                  text: str, 
                                  relation_config: RelationConfig) -> List[Dict]:
        """Extract entities using LLM with relation-specific prompt"""
        prompt = relation_config.extraction_prompt_template.format(text=text)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
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
                return []
            
            json_str = llm_output[json_start:json_end]
            entities_data = json.loads(json_str)
            
            # Parse entities using relation-specific parser
            parsed_entities = []
            for entity in entities_data:
                parsed = relation_config.entity_parser(entity)
                if parsed:
                    parsed_entities.append(parsed)
            
            return parsed_entities
            
        except Exception as e:
            print(f"    ✗ Extraction error: {e}")
            return []
    
    # ========================================================================
    # NEO4J GRAPH OPERATIONS
    # ========================================================================
    
    def create_node(self, session, node_type: str, name: str, properties: Dict = None):
        """Create or update a node"""
        props = properties or {}
        props_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
        
        query = f"""
        MERGE (n:{node_type} {{name: $name}})
        ON CREATE SET 
            n.created_at = datetime()
            {', ' + props_str if props_str else ''}
        ON MATCH SET
            {props_str if props_str else 'n.updated_at = datetime()'}
        RETURN n
        """
        
        params = {"name": name, **props}
        result = session.run(query, **params)
        return result.single()
    
    def create_relationship(self, 
                          session,
                          source_type: str,
                          source_name: str,
                          target_type: str,
                          target_name: str,
                          rel_type: str,
                          properties: Dict = None,
                          source_chunk: str = None,
                          similarity: float = None):
        """Create relationship between nodes (prevents duplicates via MERGE)"""
        props = properties or {}
        
        # Add metadata
        if source_chunk:
            props['source_chunk'] = source_chunk[:200]
        if similarity is not None:
            props['confidence'] = similarity
        
        # Build property setters for ON CREATE and ON MATCH
        create_props = []
        match_props = []
        
        for key in props.keys():
            create_props.append(f"r.{key} = ${key}")
            # For ON MATCH, update confidence if new value is higher
            if key == 'confidence':
                match_props.append(f"r.{key} = CASE WHEN ${key} > r.{key} THEN ${key} ELSE r.{key} END")
            elif key == 'source_chunk':
                # Append source chunk if not already present
                match_props.append(f"r.{key} = CASE WHEN NOT ${key} IN [r.{key}] THEN ${key} ELSE r.{key} END")
            else:
                match_props.append(f"r.{key} = ${key}")
        
        create_str = ", ".join(create_props) if create_props else ""
        match_str = ", ".join(match_props) if match_props else "r.updated_at = datetime()"
        
        query = f"""
        MATCH (s:{source_type} {{name: $source_name}})
        MATCH (t:{target_type} {{name: $target_name}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET 
            r.created_at = datetime()
            {', ' + create_str if create_str else ''}
        ON MATCH SET
            {match_str}
        RETURN r
        """
        
        params = {
            "source_name": source_name,
            "target_name": target_name,
            **props
        }
        
        result = session.run(query, **params)
        return result.single()
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def extract_relation(self, 
                        relation_name: str,
                        chunk_similarity_threshold: float = CHUNK_SIMILARITY_THRESHOLD) -> Dict:
        """
        Extract a specific relation type using hierarchical retrieval + LLM
        
        Returns statistics about extraction
        """
        # Get relation configuration
        relation_config = get_relation_config(relation_name)
        if not relation_config:
            print(f"✗ Unknown relation type: {relation_name}")
            print(f"  Available: {', '.join(list_available_relations())}")
            return {"error": "Unknown relation type"}
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING RELATION: {relation_config.name}")
        print(f"{'='*60}")
        print(f"Source: {relation_config.source_entity_type}")
        print(f"Target: {relation_config.target_entity_type}")
        print(f"Relationship: {relation_config.relationship_type}")
        
        # Step 1: Hierarchical retrieval
        chunks = self.hierarchical_retrieval(relation_config)
        
        if not chunks:
            print(f"\n✗ No relevant chunks found for {relation_name}")
            return {"relation": relation_name, "entities": 0, "relationships": 0}
        
        # Filter by chunk similarity threshold
        filtered_chunks = [c for c in chunks if c['similarity'] >= chunk_similarity_threshold]
        print(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {chunk_similarity_threshold})")
        
        # Step 2: Extract entities and build graph
        print(f"\n{'='*60}")
        print(f"EXTRACTING ENTITIES WITH LLM")
        print(f"{'='*60}")
        
        total_entities = 0
        total_relationships = 0
        
        # Track created relationships to avoid duplicates
        created_relationships = set()
        
        # For CEO extraction, track all candidates with their confidence
        ceo_candidates = []
        
        with self.driver.session() as session:
            for i, chunk in enumerate(filtered_chunks, 1):
                text = chunk['text']
                similarity = chunk['similarity']
                section = chunk['section_title']
                
                print(f"\nChunk {i}/{len(filtered_chunks)} (similarity: {similarity:.3f})")
                print(f"  Section: {section}")
                
                # Extract entities
                entities = self.extract_entities_with_llm(text, relation_config)
                
                if entities:
                    print(f"  ✓ Found {len(entities)} entities:")
                    
                    for entity in entities:
                        source = entity['source']
                        target = entity['target']
                        rel_type = entity['relationship']
                        rel_props = entity.get('properties', {})
                        
                        # Create unique key for this relationship
                        rel_key = (source['type'], source['name'], rel_type, target['type'], target['name'])
                        
                        # For CEO, collect candidates instead of creating immediately
                        if rel_type == 'CEO_OF':
                            ceo_candidates.append({
                                'entity': entity,
                                'similarity': similarity,
                                'text': text,
                                'rel_key': rel_key
                            })
                            print(f"    ~ CEO candidate: {source['name']} (confidence: {similarity:.3f})")
                            continue
                        
                        # Skip if already created
                        if rel_key in created_relationships:
                            print(f"    ⊘ Skipping duplicate: ({source['type']}: {source['name']}) "
                                  f"--[{rel_type}]--> "
                                  f"({target['type']}: {target['name']})")
                            continue
                        
                        print(f"    - ({source['type']}: {source['name']}) "
                              f"--[{rel_type}]--> "
                              f"({target['type']}: {target['name']})")
                        
                        # Create source node
                        source_props = source.get('properties', {})
                        self.create_node(session, source['type'], source['name'], source_props)
                        
                        # Create target node
                        target_props = target.get('properties', {})
                        self.create_node(session, target['type'], target['name'], target_props)
                        
                        # Create relationship
                        self.create_relationship(
                            session,
                            source['type'], source['name'],
                            target['type'], target['name'],
                            rel_type,
                            rel_props,
                            text,
                            similarity
                        )
                        
                        # Mark as created
                        created_relationships.add(rel_key)
                        
                        total_entities += 1
                        total_relationships += 1
                else:
                    print(f"    ✗ No entities extracted")
            
            # Process CEO candidates - only keep the highest confidence one
            if ceo_candidates:
                print(f"\n  Processing {len(ceo_candidates)} CEO candidates...")
                # Sort by similarity (confidence)
                ceo_candidates.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Take only the top candidate
                best_ceo = ceo_candidates[0]
                entity = best_ceo['entity']
                source = entity['source']
                target = entity['target']
                rel_type = entity['relationship']
                rel_props = entity.get('properties', {})
                
                print(f"  ✓ Selected CEO: {source['name']} (confidence: {best_ceo['similarity']:.3f})")
                
                # Create nodes and relationship
                source_props = source.get('properties', {})
                self.create_node(session, source['type'], source['name'], source_props)
                
                target_props = target.get('properties', {})
                self.create_node(session, target['type'], target['name'], target_props)
                
                self.create_relationship(
                    session,
                    source['type'], source['name'],
                    target['type'], target['name'],
                    rel_type,
                    rel_props,
                    best_ceo['text'],
                    best_ceo['similarity']
                )
                
                total_entities += 1
                total_relationships += 1
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE: {relation_name}")
        print(f"{'='*60}")
        print(f"Entities: {total_entities}")
        print(f"Relationships: {total_relationships}")
        
        return {
            "relation": relation_name,
            "entities": total_entities,
            "relationships": total_relationships
        }
    
    def extract_multiple_relations(self, relation_names: List[str]) -> Dict:
        """Extract multiple relation types"""
        print(f"\n{'='*60}")
        print(f"MULTI-RELATION EXTRACTION")
        print(f"{'='*60}")
        print(f"Relations to extract: {', '.join(relation_names)}")
        
        results = {}
        for relation_name in relation_names:
            result = self.extract_relation(relation_name)
            results[relation_name] = result
        
        # Summary
        print(f"\n{'='*60}")
        print(f"MULTI-RELATION EXTRACTION SUMMARY")
        print(f"{'='*60}")
        
        total_entities = sum(r.get('entities', 0) for r in results.values())
        total_relationships = sum(r.get('relationships', 0) for r in results.values())
        
        for relation, stats in results.items():
            print(f"{relation}: {stats.get('entities', 0)} entities, "
                  f"{stats.get('relationships', 0)} relationships")
        
        print(f"\nTotal: {total_entities} entities, {total_relationships} relationships")
        
        return results
    
    def show_graph_stats(self):
        """Display graph statistics"""
        with self.driver.session() as session:
            print(f"\n{'='*60}")
            print("KNOWLEDGE GRAPH STATISTICS")
            print(f"{'='*60}")
            
            # Node counts
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            print("\nNodes:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # Relationship counts
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            print("\nRelationships:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Relation Knowledge Graph Builder"
    )
    parser.add_argument(
        "relations",
        nargs="*",
        help=f"Relation types to extract. Available: {', '.join(list_available_relations())}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all available relation types"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available relation types"
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
        "--clear",
        action="store_true",
        help="Clear existing graph before building"
    )
    
    args = parser.parse_args()
    
    # List available relations
    if args.list:
        print("Available relation types:")
        for relation in list_available_relations():
            config = get_relation_config(relation)
            print(f"\n  {relation}")
            print(f"    {config.source_entity_type} --[{config.relationship_type}]--> {config.target_entity_type}")
            print(f"    Section keywords: {config.section_keywords}")
            print(f"    Chunk keywords: {config.chunk_keywords}")
        return
    
    # Determine which relations to extract
    if args.all:
        relations_to_extract = list_available_relations()
    elif args.relations:
        relations_to_extract = [r.upper() for r in args.relations]
    else:
        print("Error: Specify relation types or use --all")
        print(f"Available: {', '.join(list_available_relations())}")
        print("Use --list for details")
        return
    
    print("="*60)
    print("MULTI-RELATION KNOWLEDGE GRAPH BUILDER")
    print("="*60)
    
    # Initialize builder
    builder = MultiRelationKGBuilder(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.collection,
        args.db_path,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    try:
        if args.clear:
            builder.clear_database()
        
        # Extract relations
        builder.extract_multiple_relations(relations_to_extract)
        
        # Show stats
        builder.show_graph_stats()
        
        print(f"\n✓ Knowledge graph built successfully!")
        print(f"\nView graph at: http://localhost:7474")
        print(f"  Username: {args.neo4j_user}")
        print(f"  Password: {args.neo4j_password}")
        
    finally:
        builder.close()


if __name__ == "__main__":
    main()
