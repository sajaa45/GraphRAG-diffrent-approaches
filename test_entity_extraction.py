#!/usr/bin/env python3
"""
Test entity extraction pipeline
"""

from entity_extraction_pipeline import VectorStoreEntityPipeline
import time


def test_pipeline():
    """Test the complete pipeline"""
    print("="*80)
    print("ENTITY EXTRACTION PIPELINE TEST")
    print("="*80)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = VectorStoreEntityPipeline(
        vector_collection="test_collection",
        vector_db_path="./test_chroma_db",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="Lexical12345",
        use_llm=False  # Use fast rule-based extraction for testing
    )
    
    # Test queries
    test_queries = [
        "What was the revenue in 2024?",
        "Who are the key executives?",
        "What are the main products?"
    ]
    
    print(f"\nTesting with {len(test_queries)} queries...")
    
    start_time = time.time()
    results = pipeline.batch_process(test_queries)
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query']}")
        print(f"  Chunks retrieved: {result['chunks_retrieved']}")
        print(f"  Entities extracted: {result['entities_extracted']}")
        print(f"  Relationships extracted: {result['relationships_extracted']}")
        
        # Show sample entities
        if result['entities']:
            print(f"  Sample entities:")
            for entity in result['entities'][:3]:
                print(f"    - {entity['name']} ({entity['type']})")
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per query: {total_time/len(test_queries):.2f}s")
    
    # Final graph stats
    if results:
        stats = results[-1]['graph_stats']
        print(f"\nFinal Knowledge Graph:")
        print(f"  Entities: {stats['entities']}")
        print(f"  Relationships: {stats['relationships']}")
        print(f"  Source chunks: {stats['chunks']}")
    
    pipeline.close()
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
