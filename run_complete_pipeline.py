#!/usr/bin/env python3
"""
Complete pipeline: PDF → Sections → Vector Store → Query
Demonstrates the full workflow from start to finish
"""

import sys
import time
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required = {
        'chromadb': 'chromadb',
        'chonkie': 'chonkie',
        'sentence_transformers': 'sentence-transformers',
        'fitz': 'PyMuPDF'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✓ All dependencies installed\n")
    return True


def find_pdf_files():
    """Find PDF files in current directory"""
    pdf_files = list(Path('.').glob('*.pdf'))
    return pdf_files


def parse_pdf(pdf_file):
    """Parse PDF into sections"""
    print(f"Parsing PDF: {pdf_file}")
    print("-" * 60)
    
    try:
        from parse_pdf import parse_pdf_to_sections
        
        output_file = Path('output') / f"{pdf_file.stem}_sections.json"
        output_file.parent.mkdir(exist_ok=True)
        
        start = time.time()
        parse_pdf_to_sections(str(pdf_file), str(output_file))
        elapsed = time.time() - start
        
        print(f"✓ Parsed in {elapsed:.2f}s")
        print(f"✓ Saved to: {output_file}\n")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Error parsing PDF: {e}\n")
        return None


def build_vector_store(sections_file, collection_name="financial_docs"):
    """Build vector store from sections"""
    print(f"Building vector store from: {sections_file}")
    print("-" * 60)
    
    try:
        from vector_store_pipeline import process_and_store
        
        start = time.time()
        process_and_store(
            sections_file=str(sections_file),
            collection_name=collection_name,
            persist_directory="./chroma_db",
            chunk_size=512,
            similarity_threshold=0.5,
            batch_size=100,
            max_workers=4
        )
        elapsed = time.time() - start
        
        print(f"\n✓ Vector store built in {elapsed:.2f}s\n")
        return True
        
    except Exception as e:
        print(f"✗ Error building vector store: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_sample_queries(collection_name="financial_docs"):
    """Run sample queries to demonstrate the system"""
    print("Running sample queries")
    print("-" * 60)
    
    try:
        from query_vector_store import VectorStoreQuery
        
        query_interface = VectorStoreQuery(
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        # Get stats
        stats = query_interface.get_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Sections: {stats['sections']}")
        print(f"  Chunks: {stats['chunks']}")
        
        # Sample queries
        queries = [
            "revenue and financial performance",
            "risk factors and challenges",
            "future outlook and strategy"
        ]
        
        print("\nSample Queries:")
        print("=" * 60)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 60)
            
            start = time.time()
            results = query_interface.hierarchical_query(
                query,
                n_sections=2,
                n_chunks_per_section=2
            )
            elapsed = time.time() - start
            
            print(f"   Time: {elapsed:.3f}s")
            print(f"   Found: {len(results['sections'])} sections, {len(results['chunks'])} chunks")
            
            if results['chunks']:
                top = results['chunks'][0]
                print(f"\n   Top Result:")
                print(f"   Section: {top['section_title']}")
                print(f"   Page: {top['page']}")
                print(f"   Similarity: {top['similarity']:.3f}")
                print(f"   Text: {top['text'][:150]}...")
        
        print("\n✓ All queries completed successfully\n")
        return True
        
    except Exception as e:
        print(f"✗ Error running queries: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("COMPLETE PIPELINE: PDF → SECTIONS → VECTOR STORE → QUERY")
    print("="*60)
    print()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies first")
        return 1
    
    # Step 2: Find PDF files
    pdf_files = find_pdf_files()
    
    if not pdf_files:
        print("No PDF files found in current directory")
        print("Please add a PDF file and run again")
        return 1
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    print()
    
    # Use first PDF
    pdf_file = pdf_files[0]
    print(f"Using: {pdf_file}\n")
    
    # Step 3: Parse PDF
    sections_file = parse_pdf(pdf_file)
    if not sections_file:
        print("Failed to parse PDF")
        return 1
    
    # Step 4: Build vector store
    collection_name = f"{pdf_file.stem}_docs"
    if not build_vector_store(sections_file, collection_name):
        print("Failed to build vector store")
        return 1
    
    # Step 5: Run sample queries
    if not run_sample_queries(collection_name):
        print("Failed to run queries")
        return 1
    
    # Success!
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nWhat was created:")
    print(f"  1. Sections file: {sections_file}")
    print(f"  2. Vector database: ./chroma_db")
    print(f"  3. Collection: {collection_name}")
    
    print("\nNext steps:")
    print(f"  1. Query your data:")
    print(f"     python query_vector_store.py 'your query' --collection {collection_name}")
    print(f"  2. Integrate with your app:")
    print(f"     from query_vector_store import VectorStoreQuery")
    print(f"     query = VectorStoreQuery(collection_name='{collection_name}')")
    print(f"  3. Build RAG system:")
    print(f"     python rag_example.py")
    
    print("\nDocumentation:")
    print("  - QUICK_START.md - Quick reference")
    print("  - VECTOR_STORE_GUIDE.md - Complete guide")
    print("  - README_VECTOR_STORE.md - Full documentation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
