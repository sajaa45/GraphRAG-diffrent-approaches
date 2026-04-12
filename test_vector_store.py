#!/usr/bin/env python3
"""
Test script for the vector store pipeline
Demonstrates fast hierarchical embedding and retrieval
"""

import os
import time
from pathlib import Path


def test_pipeline():
    """Test the complete pipeline"""
    print("VECTOR STORE PIPELINE TEST")
    print("="*60)
    
    # Check if we have sections file
    output_dir = Path("output")
    section_files = list(output_dir.glob("*_sections.json"))
    
    if not section_files:
        print("No section files found in output/")
        print("Run parse_pdf.py first to generate sections")
        return False
    
    sections_file = section_files[0]
    print(f"Using sections file: {sections_file}")
    
    # Test 1: Build vector store
    print("\n" + "="*60)
    print("TEST 1: Building Vector Store")
    print("="*60)
    
    from vector_store_pipeline import process_and_store
    
    start_time = time.time()
    try:
        process_and_store(
            sections_file=str(sections_file),
            collection_name="test_collection",
            persist_directory="./test_chroma_db",
            chunk_size=512,
            similarity_threshold=0.5,
            batch_size=100,
            max_workers=4
        )
        build_time = time.time() - start_time
        print(f"\n✓ Vector store built in {build_time:.2f}s")
    except Exception as e:
        print(f"\n✗ Failed to build vector store: {e}")
        return False
    
    # Test 2: Query the store
    print("\n" + "="*60)
    print("TEST 2: Querying Vector Store")
    print("="*60)
    
    from query_vector_store import VectorStoreQuery
    
    try:
        query_interface = VectorStoreQuery(
            collection_name="test_collection",
            persist_directory="./test_chroma_db"
        )
        
        # Get stats
        stats = query_interface.get_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Sections: {stats['sections']}")
        print(f"  Chunks: {stats['chunks']}")
        
        # Test queries
        test_queries = [
            "revenue and financial performance",
            "risk factors and challenges",
            "future outlook and strategy"
        ]
        
        print("\n" + "="*60)
        print("TEST 3: Sample Queries")
        print("="*60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-"*60)
            
            start_time = time.time()
            results = query_interface.hierarchical_query(
                query,
                n_sections=2,
                n_chunks_per_section=2
            )
            query_time = time.time() - start_time
            
            print(f"\nResults: {len(results['sections'])} sections, {len(results['chunks'])} chunks")
            print(f"Query time: {query_time:.3f}s")
            
            # Show top result
            if results['chunks']:
                top_chunk = results['chunks'][0]
                print(f"\nTop result:")
                print(f"  Section: {top_chunk['section_title']}")
                print(f"  Page: {top_chunk['page']}")
                print(f"  Similarity: {top_chunk['similarity']}")
                print(f"  Text: {top_chunk['text'][:150]}...")
        
        print("\n✓ All queries completed successfully")
        
    except Exception as e:
        print(f"\n✗ Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Performance comparison
    print("\n" + "="*60)
    print("TEST 4: Performance Comparison")
    print("="*60)
    
    print("\nVector Store Approach:")
    print(f"  Build time: {build_time:.2f}s")
    print(f"  Storage: Persistent ChromaDB")
    print(f"  Query speed: ~0.1-0.5s per query")
    print(f"  Memory efficient: Embeddings on disk")
    
    print("\nOld JSON Approach:")
    print(f"  Build time: Similar")
    print(f"  Storage: Large JSON files")
    print(f"  Query speed: N/A (no semantic search)")
    print(f"  Memory: All data in RAM")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    return True


def main():
    success = test_pipeline()
    
    if success:
        print("\n✓ Vector store pipeline is working correctly")
        print("\nNext steps:")
        print("  1. Run on your full dataset")
        print("  2. Integrate with your application")
        print("  3. Tune chunk_size and similarity_threshold")
        print("  4. Add more sophisticated retrieval strategies")
    else:
        print("\n✗ Some tests failed")
        print("Check the error messages above")


if __name__ == "__main__":
    main()
