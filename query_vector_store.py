#!/usr/bin/env python3
"""
Query interface for the hierarchical vector store with keyword-based section discovery
"""

# ============================================================================
# QUERY CONFIGURATION - Edit this to change your query
# ============================================================================
DEFAULT_QUERY = "CEO"  # Change this to your query
SECTION_THRESHOLD = 0.4  # Minimum similarity for section-based search (0-1)
# ============================================================================

import argparse
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStoreQuery:
    """Query interface for hierarchical vector store"""
    
    def __init__(self, 
                 collection_name: str = "financial_docs",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize query interface"""
        print(f"Loading vector store: {collection_name}")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Connected to collection: {collection_name}")
            print(f"Total documents: {self.collection.count()}")
        except Exception as e:
            print(f"Error: Collection '{collection_name}' not found")
            raise e
    
    def hierarchical_query(self, 
                          query_text: str,
                          n_sections: int = 2,
                          n_chunks_per_section: int = 3,
                          section_threshold: float = 0.4) -> Dict:
        """
        Hierarchical search with keyword-based section discovery:
        1. For queries like "CEO", search sections using keywords ("overview board governance")
        2. If relevant sections found, search for original query in chunks within them
        3. Otherwise fall back to direct chunk search
        """
        print(f"\nHierarchical Query: '{query_text}'")
        print("="*60)
        
        # Map queries to section-finding keywords
        section_keywords_map = {
            'ceo': 'overview board governance directors leadership management executive',
            'cfo': 'overview board governance directors leadership management executive',
            'chief executive': 'overview board governance directors leadership',
            'board': 'board governance directors corporate',
            'director': 'board governance directors corporate',
            'revenue': 'results performance financial operations',
            'profit': 'results performance financial operations earnings',
            'risk': 'risk factors uncertainties management',
            'sustainability': 'sustainability environmental esg social',
        }
        
        query_lower = query_text.lower()
        section_search_keywords = None
        
        # Find matching keywords for section search
        for term, keywords in section_keywords_map.items():
            if term in query_lower:
                section_search_keywords = keywords
                break
        
        # Embed queries
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        use_sections = False
        section_results = None
        
        # Step 1: Search sections using keywords (if applicable)
        if section_search_keywords:
            print(f"\nStep 1: Searching sections with keywords: '{section_search_keywords}'")
            section_embedding = self.embedding_model.encode([section_search_keywords])[0]
            
            section_results = self.collection.query(
                query_embeddings=[section_embedding.tolist()],
                n_results=n_sections,
                where={"type": "section"}
            )
            
            if section_results['documents'][0]:
                best_similarity = 1 - section_results['distances'][0][0]
                print(f"  Best section similarity: {best_similarity:.3f}")
                
                if best_similarity >= section_threshold:
                    use_sections = True
                    print(f"  ✓ Found {len(section_results['documents'][0])} relevant sections")
                    for meta in section_results['metadatas'][0]:
                        print(f"    - {meta['title']}")
                else:
                    print(f"  ✗ Below threshold ({best_similarity:.3f} < {section_threshold})")
        
        # Step 2: If no keyword match, try direct section search
        if not use_sections and not section_search_keywords:
            print(f"\nStep 1: Searching sections directly with query")
            section_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_sections,
                where={"type": "section"}
            )
            
            if section_results['documents'][0]:
                best_similarity = 1 - section_results['distances'][0][0]
                if best_similarity >= section_threshold:
                    use_sections = True
                    print(f"  ✓ Found sections (similarity: {best_similarity:.3f})")
        
        results = {
            "query": query_text,
            "search_mode": "hierarchical" if use_sections else "direct_chunks",
            "sections": [],
            "chunks": []
        }
        
        if use_sections:
            # Search chunks within relevant sections using ORIGINAL query
            print(f"\nStep 2: Searching for '{query_text}' in chunks within sections...")
            
            for i, (doc, meta, distance) in enumerate(zip(
                section_results['documents'][0],
                section_results['metadatas'][0],
                section_results['distances'][0]
            ), 1):
                section_id = meta['section_id']
                section_title = meta['title']
                
                print(f"\n  Section {i}: {section_title} (Pages {meta['start_page']}-{meta['end_page']})")
                
                results['sections'].append({
                    "title": section_title,
                    "pages": f"{meta['start_page']}-{meta['end_page']}",
                    "similarity": round(1 - distance, 3)
                })
                
                # Search chunks in this section with ORIGINAL query
                chunk_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
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
                        
                        chunk_data = {
                            "section_title": section_title,
                            "section_id": section_id,
                            "chunk_index": chunk_meta.get('chunk_index', j),
                            "similarity": round(chunk_similarity, 3),
                            "text": chunk_doc
                        }
                        
                        # Add page information if available
                        if 'source_page' in chunk_meta:
                            chunk_data['source_page'] = chunk_meta['source_page']
                            chunk_data['section_page_range'] = chunk_meta.get('section_page_range')
                        elif 'start_page' in chunk_meta:
                            chunk_data['start_page'] = chunk_meta['start_page']
                            chunk_data['end_page'] = chunk_meta.get('end_page')
                        
                        results['chunks'].append(chunk_data)
        else:
            # Fallback: Direct chunk search
            print(f"\nStep 2: Direct chunk search (no relevant sections)")
            
            n_chunks = n_sections * n_chunks_per_section
            chunk_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_chunks,
                where={"type": "chunk"}
            )
            
            if chunk_results['documents'][0]:
                from collections import defaultdict
                sections_map = defaultdict(list)
                
                for doc, meta, distance in zip(
                    chunk_results['documents'][0],
                    chunk_results['metadatas'][0],
                    chunk_results['distances'][0]
                ):
                    section_title = meta.get('section_title', 'Unknown')
                    sections_map[section_title].append({
                        'text': doc,
                        'metadata': meta,
                        'similarity': 1 - distance
                    })
                
                for section_title, chunks in sections_map.items():
                    meta = chunks[0]['metadata']
                    results['sections'].append({
                        "title": section_title,
                        "pages": f"{meta.get('start_page', '?')}-{meta.get('end_page', '?')}",
                        "best_chunk_similarity": round(max(c['similarity'] for c in chunks), 3)
                    })
                    
                    for chunk in chunks:
                        chunk_data = {
                            "section_title": section_title,
                            "section_id": chunk['metadata'].get('section_id'),
                            "chunk_index": chunk['metadata'].get('chunk_index'),
                            "similarity": round(chunk['similarity'], 3),
                            "text": chunk['text']
                        }
                        
                        # Add page information if available
                        meta = chunk['metadata']
                        if 'source_page' in meta:
                            chunk_data['source_page'] = meta['source_page']
                            chunk_data['section_page_range'] = meta.get('section_page_range')
                        elif 'start_page' in meta:
                            chunk_data['start_page'] = meta['start_page']
                            chunk_data['end_page'] = meta.get('end_page')
                        
                        results['chunks'].append(chunk_data)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        total = self.collection.count()
        sections = self.collection.get(where={"type": "section"}, limit=10000)
        chunks = self.collection.get(where={"type": "chunk"}, limit=10000)
        
        return {
            "total_documents": total,
            "sections": len(sections['ids']),
            "chunks": len(chunks['ids'])
        }


def main():
    parser = argparse.ArgumentParser(description="Query the hierarchical vector store")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help=f"Search query (default: '{DEFAULT_QUERY}')")
    parser.add_argument("--collection", "-c", default="financial_docs", help="Collection name")
    parser.add_argument("--db-path", "-d", default="./chroma_db", help="Database path")
    parser.add_argument("--section-threshold", type=float, default=SECTION_THRESHOLD, help=f"Section threshold (default: {SECTION_THRESHOLD})")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    try:
        query_interface = VectorStoreQuery(
            collection_name=args.collection,
            persist_directory=args.db_path
        )
    except Exception as e:
        print(f"\nError: {e}")
        return
    
    if args.stats:
        print("\nDatabase Statistics:")
        print("="*60)
        stats = query_interface.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
    
    if args.query:
        results = query_interface.hierarchical_query(
            args.query,
            n_sections=2,
            n_chunks_per_section=3,
            section_threshold=args.section_threshold
        )
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Search mode: {results['search_mode']}")
        print(f"Found {len(results['sections'])} sections with {len(results['chunks'])} chunks")
        
        # Save to file
        if results['chunks']:
            import time
            output_file = f"output/query_results_{int(time.time())}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"QUERY RESULTS: '{results['query']}'\n")
                f.write("="*80 + "\n\n")
                f.write(f"Search mode: {results['search_mode']}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total sections: {len(results['sections'])}\n")
                f.write(f"Total chunks: {len(results['chunks'])}\n\n")
                
                f.write("="*80 + "\n")
                f.write("SECTIONS FOUND\n")
                f.write("="*80 + "\n\n")
                for i, section in enumerate(results['sections'], 1):
                    f.write(f"{i}. {section['title']}\n")
                    f.write(f"   Pages: {section['pages']}\n")
                    if 'similarity' in section:
                        f.write(f"   Similarity: {section['similarity']:.3f}\n")
                    elif 'best_chunk_similarity' in section:
                        f.write(f"   Best chunk similarity: {section['best_chunk_similarity']:.3f}\n")
                    f.write("\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("MATCHED CHUNKS (Ordered by Similarity)\n")
                f.write("="*80 + "\n\n")
                
                for i, chunk in enumerate(results['chunks'], 1):
                    f.write(f"CHUNK {i}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Section: {chunk['section_title']}\n")
                    f.write(f"Similarity: {chunk['similarity']:.3f}\n")
                    f.write(f"Chunk Index: {chunk.get('chunk_index', '?')}\n")
                    f.write(f"Length: {len(chunk['text'])} characters\n")
                    f.write("-"*80 + "\n")
                    f.write(chunk['text'])
                    f.write("\n\n" + "="*80 + "\n\n")
            
            print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
