#!/usr/bin/env python3
"""
Fast vector store pipeline with hierarchical embeddings
Uses existing LlamaIndex chunking system and stores in ChromaDB
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Use existing chunking system
from chunking import llamaindex_chunker, get_embedding_model

# Vector store
import chromadb
from chromadb.config import Settings


class HierarchicalVectorStore:
    """Manages hierarchical embeddings in ChromaDB with fast parallel processing"""
    
    def __init__(self, 
                 collection_name: str = "financial_docs",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize vector store
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Where to store the database
        """
        print(f"Initializing vector store: {collection_name}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
    
    def add_documents_batch(self, 
                           documents: List[str],
                           metadatas: List[Dict],
                           ids: List[str],
                           embeddings: List[List[float]],
                           batch_size: int = 100):
        """
        Add documents in batches with pre-computed embeddings
        
        Args:
            documents: List of text documents
            metadatas: List of metadata dicts
            ids: List of unique IDs
            embeddings: Pre-computed embeddings
            batch_size: Size of batches for processing
        """
        total = len(documents)
        print(f"Adding {total} documents in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_emb = embeddings[i:i+batch_size]
            
            # Add to collection
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
                embeddings=batch_emb
            )
            
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= total:
                print(f"  Progress: {min(i+batch_size, total)}/{total}")
    
    def query(self, query_text: str, n_results: int = 5, 
              filter_dict: Dict = None) -> Dict:
        """
        Query the vector store
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            filter_dict: Optional metadata filter
        """
        from sentence_transformers import SentenceTransformer
        
        # Load model for query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query_text])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict
        )
        
        return results


def process_and_store(sections_file: str,
                     collection_name: str = "financial_docs",
                     persist_directory: str = "./chroma_db",
                     chunk_size: int = 512,
                     similarity_threshold: float = 0.5,
                     batch_size: int = 100,
                     buffer_size: int = 1,
                     threshold: int = 70,
                     use_existing_chunks: bool = True):
    """
    Main pipeline: process sections using existing chunking system and store in vector database
    
    Args:
        sections_file: Path to sections JSON file
        collection_name: ChromaDB collection name
        persist_directory: Where to persist the database
        chunk_size: Target chunk size (not used with LlamaIndex, kept for compatibility)
        similarity_threshold: Semantic similarity threshold (not used with LlamaIndex)
        batch_size: Batch size for storing
        buffer_size: LlamaIndex buffer size
        threshold: LlamaIndex threshold
        use_existing_chunks: Try to load pre-computed chunks if available
    """
    start_time = time.time()
    
    print("="*60)
    print("HIERARCHICAL VECTOR STORE PIPELINE")
    print("Using existing LlamaIndex chunking system")
    print("="*60)
    
    # Check for existing chunks file
    output_dir = Path(sections_file).parent
    existing_chunks_file = output_dir / "SemanticSplitterNodeParser_chunks.json"
    
    all_chunk_docs = []
    
    if use_existing_chunks and existing_chunks_file.exists():
        print(f"\n✓ Found existing chunks file: {existing_chunks_file}")
        print("Loading pre-computed chunks with embeddings...")
        
        try:
            with open(existing_chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = chunks_data.get('chunks', [])
            if chunks and len(chunks) > 0:
                print(f"✓ Loaded {len(chunks)} pre-computed chunks with embeddings")
                
                # Convert to our format
                for i, chunk in enumerate(chunks, 1):
                    chunk_doc = {
                        "id": f"chunk_{i}",
                        "text": chunk['text'],
                        "metadata": {
                            "type": "chunk",
                            "chunk_id": i,
                            "page": chunk.get('source_page', chunk.get('page_range', 'unknown')),
                            "section_title": chunk.get('section_path', 'unknown'),
                            "section_level": chunk.get('section_level', 0),
                            "char_count": chunk.get('length', len(chunk['text']))
                        },
                        "embedding": chunk.get('embedding', [])
                    }
                    all_chunk_docs.append(chunk_doc)
                
                print(f"✓ Converted {len(all_chunk_docs)} chunks to vector store format")
                
                # Skip to storage
                chunks_loaded = True
            else:
                print("⚠ Chunks file is empty, will generate new chunks")
                chunks_loaded = False
        except Exception as e:
            print(f"⚠ Error loading existing chunks: {e}")
            print("Will generate new chunks instead")
            chunks_loaded = False
    else:
        if use_existing_chunks:
            print(f"\n⚠ No existing chunks file found at: {existing_chunks_file}")
        print("Will generate chunks from sections...")
        chunks_loaded = False
    
    # If we didn't load existing chunks, generate them
    if not chunks_loaded:
        # Load sections
        print(f"\nLoading sections from: {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', [])
        if not sections:
            print("No sections found!")
            return
        
        print(f"Found {len(sections)} top-level sections")
        
        # Initialize embedding model
        embed_model = get_embedding_model()
        if not embed_model:
            print("Error: Could not load embedding model")
            return
        
        # Process all sections using existing chunking system
        print("\nProcessing sections with LlamaIndex chunker...")
        all_section_docs = []
        chunk_id = 1
        
        for i, section in enumerate(sections, 1):
            print(f"  Section {i}/{len(sections)}: {section.get('title', 'Untitled')}")
            
            # Create section document
            section_doc = {
                "id": f"section_{section.get('id', i)}",
                "text": section.get('text', ''),
                "metadata": {
                    "type": "section",
                    "title": section['title'],
                    "level": section.get('level', 0),
                    "start_page": section.get('start_page', 0),
                    "end_page": section.get('end_page', 0)
                },
                "embedding": embed_model.encode([section.get('text', '')])[0].tolist() if section.get('text') else []
            }
            all_section_docs.append(section_doc)
            
            # Chunk section text using existing LlamaIndex chunker
            section_text = section.get('text', '')
            if section_text and section_text.strip():
                filtered_chunks, stats = llamaindex_chunker(
                    section_text,
                    buffer_size=buffer_size,
                    threshold=threshold,
                    embed_model=embed_model,
                    debug=False
                )
                
                for chunk_data in filtered_chunks:
                    chunk_doc = {
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_data['text'],
                        "metadata": {
                            "type": "chunk",
                            "chunk_id": chunk_id,
                            "page": section.get('start_page', 0),
                            "section_title": section['title'],
                            "section_level": section.get('level', 0),
                            "char_count": len(chunk_data['text'])
                        },
                        "embedding": chunk_data['embedding']
                    }
                    all_chunk_docs.append(chunk_doc)
                    chunk_id += 1
            
            # Process subsections recursively if they exist
            if 'subsections' in section and section['subsections']:
                for subsection in section['subsections']:
                    sections.append(subsection)
        
        print(f"\nGenerated:")
        print(f"  {len(all_section_docs)} section embeddings")
        print(f"  {len(all_chunk_docs)} chunk embeddings")
        
        # Store sections first
        print("\nStoring section embeddings...")
        section_texts = [doc['text'] for doc in all_section_docs]
        section_metas = [doc['metadata'] for doc in all_section_docs]
        section_ids = [doc['id'] for doc in all_section_docs]
        section_embs = [doc['embedding'] for doc in all_section_docs]
        
        vector_store = HierarchicalVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        vector_store.add_documents_batch(
            documents=section_texts,
            metadatas=section_metas,
            ids=section_ids,
            embeddings=section_embs,
            batch_size=batch_size
        )
    else:
        # Just initialize vector store for chunks
        vector_store = HierarchicalVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    # Store chunks (whether loaded or generated)
    print("\nStoring chunk embeddings...")
    chunk_texts = [doc['text'] for doc in all_chunk_docs]
    chunk_metas = [doc['metadata'] for doc in all_chunk_docs]
    chunk_ids = [doc['id'] for doc in all_chunk_docs]
    chunk_embs = [doc['embedding'] for doc in all_chunk_docs]
    
    vector_store.add_documents_batch(
        documents=chunk_texts,
        metadatas=chunk_metas,
        ids=chunk_ids,
        embeddings=chunk_embs,
        batch_size=batch_size
    )
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Database location: {persist_directory}")
    print(f"Collection: {collection_name}")
    print(f"\nStatistics:")
    print(f"  Total chunks: {len(all_chunk_docs)}")
    if not chunks_loaded:
        print(f"  Sections: {len(all_section_docs)}")
    print(f"  Processing speed: {len(all_chunk_docs)/total_time:.1f} docs/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Process sections and store embeddings in vector database using existing LlamaIndex chunker"
    )
    parser.add_argument(
        "sections_file",
        help="Path to sections JSON file"
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial_docs",
        help="ChromaDB collection name (default: financial_docs)"
    )
    parser.add_argument(
        "--db-path", "-d",
        default="./chroma_db",
        help="Database persist directory (default: ./chroma_db)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for storing (default: 100)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1,
        help="LlamaIndex buffer size (default: 1)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=70,
        help="LlamaIndex threshold (default: 70)"
    )
    
    args = parser.parse_args()
    
    process_and_store(
        sections_file=args.sections_file,
        collection_name=args.collection,
        persist_directory=args.db_path,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()

