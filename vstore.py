#!/usr/bin/env python3
"""
Simple CLI tool for vector store operations
Usage: python vstore.py <command> [options]
"""

import sys
import argparse
from pathlib import Path


def cmd_build(args):
    """Build vector store from sections file"""
    from vector_store_pipeline import process_and_store
    
    if not Path(args.sections_file).exists():
        print(f"Error: File not found: {args.sections_file}")
        return 1
    
    print(f"Building vector store from: {args.sections_file}")
    
    try:
        process_and_store(
            sections_file=args.sections_file,
            collection_name=args.collection,
            persist_directory=args.db_path,
            chunk_size=args.chunk_size,
            similarity_threshold=args.threshold,
            batch_size=args.batch_size,
            max_workers=args.workers
        )
        print("\n✓ Vector store built successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def cmd_query(args):
    """Query the vector store"""
    from query_vector_store import VectorStoreQuery
    
    try:
        query_interface = VectorStoreQuery(
            collection_name=args.collection,
            persist_directory=args.db_path
        )
        
        if args.mode == "hierarchical":
            results = query_interface.hierarchical_query(
                args.query,
                n_sections=args.n_sections,
                n_chunks_per_section=args.n_chunks
            )
            
            print(f"\nFound {len(results['sections'])} sections with {len(results['chunks'])} chunks\n")
            
            for i, chunk in enumerate(results['chunks'][:args.limit], 1):
                print(f"{i}. [{chunk['section_title']}] Page {chunk['page']}")
                print(f"   Similarity: {chunk['similarity']:.3f}")
                print(f"   {chunk['text'][:200]}...")
                print()
        
        elif args.mode == "chunks":
            results = query_interface.query_chunks(
                args.query,
                n_results=args.limit,
                page=args.page
            )
            
            print(f"\nFound {len(results['documents'][0])} chunks\n")
            
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"{i}. Page {meta['page']} - {meta['section_title']}")
                print(f"   Similarity: {1-dist:.3f}")
                print(f"   {doc[:200]}...")
                print()
        
        elif args.mode == "sections":
            results = query_interface.query_sections(
                args.query,
                n_results=args.limit
            )
            
            print(f"\nFound {len(results['documents'][0])} sections\n")
            
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"{i}. {meta['title']}")
                print(f"   Path: {meta['path']}")
                print(f"   Pages: {meta['start_page']}-{meta['end_page']}")
                print(f"   Similarity: {1-dist:.3f}")
                print()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def cmd_stats(args):
    """Show database statistics"""
    from query_vector_store import VectorStoreQuery
    
    try:
        query_interface = VectorStoreQuery(
            collection_name=args.collection,
            persist_directory=args.db_path
        )
        
        stats = query_interface.get_stats()
        
        print("\nDatabase Statistics")
        print("=" * 60)
        print(f"Collection: {args.collection}")
        print(f"Location: {args.db_path}")
        print(f"\nDocuments:")
        print(f"  Total: {stats['total_documents']}")
        print(f"  Sections: {stats['sections']}")
        print(f"  Chunks: {stats['chunks']}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def cmd_list(args):
    """List available collections"""
    import chromadb
    from chromadb.config import Settings
    
    try:
        client = chromadb.PersistentClient(
            path=args.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        
        print("\nAvailable Collections")
        print("=" * 60)
        
        if not collections:
            print("No collections found")
        else:
            for coll in collections:
                print(f"  - {coll.name} ({coll.count()} documents)")
        
        print()
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Vector Store CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build vector store
  python vstore.py build output/sections.json
  
  # Query
  python vstore.py query "revenue growth"
  
  # Show stats
  python vstore.py stats
  
  # List collections
  python vstore.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build vector store')
    build_parser.add_argument('sections_file', help='Path to sections JSON file')
    build_parser.add_argument('--collection', '-c', default='financial_docs', help='Collection name')
    build_parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    build_parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size')
    build_parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    build_parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    build_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query vector store')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--collection', '-c', default='financial_docs', help='Collection name')
    query_parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    query_parser.add_argument('--mode', choices=['hierarchical', 'chunks', 'sections'], 
                             default='hierarchical', help='Query mode')
    query_parser.add_argument('--limit', '-n', type=int, default=5, help='Number of results')
    query_parser.add_argument('--page', type=int, help='Filter by page')
    query_parser.add_argument('--n-sections', type=int, default=2, help='Number of sections (hierarchical mode)')
    query_parser.add_argument('--n-chunks', type=int, default=3, help='Chunks per section (hierarchical mode)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--collection', '-c', default='financial_docs', help='Collection name')
    stats_parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List collections')
    list_parser.add_argument('--db-path', '-d', default='./chroma_db', help='Database path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command handler
    commands = {
        'build': cmd_build,
        'query': cmd_query,
        'stats': cmd_stats,
        'list': cmd_list
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
