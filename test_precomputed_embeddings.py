#!/usr/bin/env python3
"""
Test script to verify that pre-computed embeddings are being used correctly
"""

import json
import numpy as np
from graphrag_system import GraphRAGSystem

def test_precomputed_embeddings():
    """Test that the system uses pre-computed embeddings from chunks"""
    
    print("="*80)
    print("Testing Pre-computed Embeddings in GraphRAG System")
    print("="*80)
    
    # Initialize GraphRAG system
    try:
        graphrag = GraphRAGSystem(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            ollama_url="http://localhost:11434"
        )
        print("\n✓ GraphRAG system initialized successfully")
    except Exception as e:
        print(f"\n✗ Failed to initialize GraphRAG system: {e}")
        return
    
    # Test query
    test_query = "What is the company's revenue?"
    
    print(f"\n{'='*80}")
    print(f"Test Query: {test_query}")
    print(f"{'='*80}")
    
    # Embed the query (only the query needs to be embedded)
    print("\n1. Embedding query...")
    query_embedding = graphrag.embed_query(test_query)
    print(f"   ✓ Query embedded: {len(query_embedding)} dimensions")
    
    # Find similar chunks using pre-computed embeddings
    print("\n2. Finding similar chunks using pre-computed embeddings...")
    print("   (This should NOT re-calculate chunk embeddings)")
    
    similar_chunks = graphrag.find_similar_chunks(query_embedding, top_k=5)
    
    if similar_chunks:
        print(f"\n   ✓ Found {len(similar_chunks)} similar chunks")
        print("\n   Top 3 Results:")
        for i, chunk in enumerate(similar_chunks[:3], 1):
            print(f"\n   Chunk {i}:")
            print(f"   - ID: {chunk['chunk_id']}")
            print(f"   - Similarity: {chunk['similarity']:.4f}")
            print(f"   - Section: {chunk['section_id']}")
            print(f"   - Pages: {chunk['page_range']}")
            print(f"   - Content preview: {chunk['content'][:100]}...")
    else:
        print("\n   ✗ No chunks found (check if embeddings are stored in Neo4j)")
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")
    
    # Verify embeddings are in JSON file
    print("\n3. Verifying embeddings in JSON file...")
    try:
        json_file = "output/SemanticSplitterNodeParser_chunks.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks_with_embeddings = sum(1 for chunk in data.get('chunks', []) 
                                     if chunk.get('embedding') and len(chunk['embedding']) > 0)
        total_chunks = len(data.get('chunks', []))
        
        print(f"   ✓ JSON file: {json_file}")
        print(f"   ✓ Total chunks: {total_chunks}")
        print(f"   ✓ Chunks with embeddings: {chunks_with_embeddings}")
        
        if chunks_with_embeddings == 0:
            print("\n   ⚠ WARNING: No embeddings found in JSON file!")
            print("   Run chunking.py first to generate chunks with embeddings")
        elif chunks_with_embeddings < total_chunks:
            print(f"\n   ⚠ WARNING: Only {chunks_with_embeddings}/{total_chunks} chunks have embeddings")
        else:
            print(f"\n   ✓ All chunks have embeddings!")
            
    except FileNotFoundError:
        print(f"   ✗ JSON file not found: {json_file}")
        print("   Run chunking.py first to generate chunks")
    except Exception as e:
        print(f"   ✗ Error reading JSON file: {e}")

if __name__ == "__main__":
    test_precomputed_embeddings()
