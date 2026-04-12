#!/usr/bin/env python3
"""
Unified Pipeline: Chunk + Store in Vector DB Simultaneously
No intermediate JSON files with embeddings - goes straight to ChromaDB
Uses the production chunking system from chunking.py
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import os

# Import chunking functions from chunking.py
from chunking import llamaindex_chunker, get_embedding_model

# Vector store
import chromadb
from chromadb.config import Settings


class UnifiedPipeline:
    """Unified pipeline that chunks and stores simultaneously"""
    
    def __init__(self,
                 collection_name: str = "financial_docs",
                 persist_directory: str = "./chroma_db",
                 buffer_size: int = 1,
                 threshold: int = 70):
        """
        Initialize unified pipeline
        
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Where to store the database
            buffer_size: LlamaIndex buffer size
            threshold: LlamaIndex threshold
        """
        print("Initializing Unified Pipeline...")
        print(f"  Collection: {collection_name}")
        print(f"  Database: {persist_directory}")
        
        # Store chunking parameters
        self.buffer_size = buffer_size
        self.threshold = threshold
        
        # Initialize embedding model (from chunking.py)
        print("  Loading embedding model...")
        self.embed_model = get_embedding_model()
        if self.embed_model:
            print("  ✓ Embedding model loaded")
        else:
            print("  ✗ Failed to load embedding model")
            raise Exception("Embedding model required for pipeline")
        
        # Initialize ChromaDB
        print("  Connecting to ChromaDB...")
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
            print(f"  ✓ Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"  ✓ Created new collection: {collection_name}")
        
        print("✓ Pipeline initialized\n")
    
    def process_section(self, section: Dict, section_idx: int, total_sections: int):
        """
        Process a single section: chunk using chunking.py and store immediately
        Also stores section-level embedding for hierarchical search
        
        Args:
            section: Section dictionary
            section_idx: Current section index
            total_sections: Total number of sections
        """
        section_title = section.get('title', 'Untitled')
        section_text = section.get('text', '')
        
        print(f"[{section_idx}/{total_sections}] Processing: {section_title}")
        
        if not section_text or not section_text.strip():
            print(f"  ⚠ Empty section, skipping")
            return 0
        
        # First, store section-level embedding for hierarchical search
        section_id = f"section_{section_idx}"
        section_embedding = self.embed_model.encode([section_text])[0].tolist()
        
        self.collection.add(
            documents=[section_text[:1000]],  # Store preview of section text
            ids=[section_id],
            metadatas=[{
                "type": "section",
                "section_id": section_idx,
                "title": section_title,
                "level": section.get('level', 0),
                "start_page": section.get('start_page', 0),
                "end_page": section.get('end_page', 0),
                "char_count": len(section_text)
            }],
            embeddings=[section_embedding]
        )
        
        # Then chunk using the production chunking system from chunking.py
        # This includes all filtering (small, decorative, TOC, meaningless, repetitive)
        filtered_chunks, filtering_stats = llamaindex_chunker(
            text=section_text,
            buffer_size=self.buffer_size,
            threshold=self.threshold,
            embed_model=self.embed_model,
            debug=False
        )
        
        if not filtered_chunks:
            print(f"  ⚠ No chunks after filtering")
            return 0
        
        # Show filtering stats
        total_filtered = sum(filtering_stats.values())
        if total_filtered > 0:
            print(f"  Filtered: {total_filtered} chunks (small:{filtering_stats.get('small',0)}, "
                  f"decorative:{filtering_stats.get('decorative',0)}, "
                  f"TOC:{filtering_stats.get('toc',0)}, "
                  f"meaningless:{filtering_stats.get('meaningless',0)}, "
                  f"repetitive:{filtering_stats.get('repetitive',0)})")
        
        # Prepare data for ChromaDB
        chunk_texts = []
        chunk_ids = []
        chunk_metadatas = []
        chunk_embeddings = []
        
        for i, chunk_data in enumerate(filtered_chunks, 1):
            chunk_id = f"section_{section_idx}_chunk_{i}"
            
            chunk_texts.append(chunk_data['text'])
            chunk_ids.append(chunk_id)
            chunk_metadatas.append({
                "type": "chunk",
                "section_id": section_idx,
                "section_title": section_title,
                "section_level": section.get('level', 0),
                "start_page": section.get('start_page', 0),
                "end_page": section.get('end_page', 0),
                "chunk_index": i,
                "total_chunks": len(filtered_chunks),
                "char_count": len(chunk_data['text'])
            })
            chunk_embeddings.append(chunk_data['embedding'])
        
        # Store directly in ChromaDB (no JSON intermediate)
        self.collection.add(
            documents=chunk_texts,
            ids=chunk_ids,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings
        )
        
        print(f"  ✓ Stored section embedding + {len(filtered_chunks)} filtered chunks in vector DB")
        return len(filtered_chunks)
    
    def process_sections_file(self, sections_file: str, save_metadata: bool = True, save_chunks: bool = True):
        """
        Process entire sections file
        
        Args:
            sections_file: Path to sections JSON file
            save_metadata: Save lightweight metadata JSON (without embeddings)
            save_chunks: Save detailed chunk files (JSON and TXT without embeddings)
        """
        start_time = time.time()
        
        print("="*70)
        print("UNIFIED PIPELINE: CHUNK → VECTOR STORE")
        print("="*70)
        
        # Load sections
        print(f"\nLoading sections from: {sections_file}")
        with open(sections_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', [])
        if not sections:
            print("✗ No sections found!")
            return
        
        print(f"✓ Found {len(sections)} sections")
        
        # Check if sections have text, if not, load from source
        has_text = any(section.get('text') for section in sections)
        
        if not has_text:
            print("⚠ Sections don't have text content, loading from source...")
            
            # Get source filename from sections data
            source_filename = data.get('filename', '')
            if source_filename:
                # Try to find the source JSON file
                base_name = os.path.splitext(os.path.basename(source_filename))[0]
                
                # Try multiple locations
                possible_paths = [
                    Path(sections_file).parent.parent / f"{base_name}.json",  # ../filename.json
                    Path(sections_file).parent / f"{base_name}.json",  # output/filename.json
                    Path("/app/input") / f"{base_name}.json",  # /app/input/filename.json (Docker)
                    Path(".") / f"{base_name}.json",  # ./filename.json
                ]
                
                source_json = None
                for path in possible_paths:
                    if path.exists():
                        source_json = path
                        break
                
                if source_json:
                    print(f"✓ Loading page text from: {source_json}")
                    with open(source_json, 'r', encoding='utf-8') as f:
                        pages_data = json.load(f)
                    
                    # Add text to sections from pages
                    sections = self._add_text_to_sections(sections, pages_data)
                    print(f"✓ Added text content to {len(sections)} sections")
                else:
                    print(f"✗ Could not find source file: {base_name}.json")
                    print(f"  Tried locations:")
                    for path in possible_paths:
                        print(f"    - {path}")
                    print("✗ Cannot proceed without text content")
                    return
            else:
                print("✗ No source filename in sections file")
                print("✗ Cannot proceed without text content")
                return
        
        print()
        
        # Process each section
        total_chunks = 0
        metadata_records = []
        all_chunk_details = []  # For detailed chunk files
        cumulative_filtering_stats = {'small': 0, 'toc': 0, 'meaningless': 0, 'repetitive': 0, 'decorative': 0}
        
        for idx, section in enumerate(sections, 1):
            chunks_count = self.process_section(section, idx, len(sections))
            total_chunks += chunks_count
            
            # Save lightweight metadata (no embeddings)
            if save_metadata:
                metadata_records.append({
                    "section_id": idx,
                    "title": section.get('title', 'Untitled'),
                    "level": section.get('level', 0),
                    "start_page": section.get('start_page', 0),
                    "end_page": section.get('end_page', 0),
                    "chunks_count": chunks_count
                })
            
            # Collect detailed chunk info for chunk files (if enabled)
            if save_chunks and chunks_count > 0:
                # Get only the chunks (not the section) we just stored from ChromaDB
                section_chunks = self.collection.get(
                    where={"$and": [{"section_id": idx}, {"type": "chunk"}]},
                    include=["documents", "metadatas"]
                )
                
                for i, (doc, meta) in enumerate(zip(section_chunks['documents'], section_chunks['metadatas'])):
                    all_chunk_details.append({
                        'text': doc,
                        'length': len(doc),
                        'section_id': meta['section_id'],
                        'section_title': meta['section_title'],
                        'section_level': meta['section_level'],
                        'start_page': meta['start_page'],
                        'end_page': meta['end_page'],
                        'chunk_index': meta['chunk_index'],
                        'total_chunks': meta['total_chunks'],
                        'method': 'SemanticSplitterNodeParser'
                    })
        
        # Save lightweight metadata file (no embeddings!)
        if save_metadata and metadata_records:
            output_dir = Path(sections_file).parent
            metadata_file = output_dir / "chunks_metadata.json"
            
            metadata = {
                "source_file": str(sections_file),
                "total_sections": len(sections),
                "total_chunks": total_chunks,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "sections": metadata_records
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✓ Saved lightweight metadata to: {metadata_file}")
            print(f"  (No embeddings in JSON - they're in the vector DB)")
        
        # Save detailed chunk files (JSON and TXT without embeddings)
        if save_chunks and all_chunk_details:
            output_dir = Path(sections_file).parent
            
            # Save JSON file (without embeddings)
            chunks_json_file = output_dir / "SemanticSplitterNodeParser_chunks.json"
            chunks_data = {
                "method": "SemanticSplitterNodeParser",
                "status": "completed",
                "total_sections": len(sections),
                "total_chunks": len(all_chunk_details),
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "chunks": all_chunk_details
            }
            
            with open(chunks_json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved detailed chunks JSON to: {chunks_json_file}")
            print(f"  (No embeddings - for human review)")
            
            # Save TXT file (human-readable)
            chunks_txt_file = output_dir / "SemanticSplitterNodeParser_chunks.txt"
            
            with open(chunks_txt_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SemanticSplitterNodeParser Chunks - Detailed View\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total sections: {len(sections)}\n")
                f.write(f"Total chunks: {len(all_chunk_details)}\n")
                f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Embeddings stored in: ChromaDB vector store\n")
                f.write("=" * 80 + "\n\n")
                
                for i, chunk in enumerate(all_chunk_details, 1):
                    f.write(f"CHUNK {i:04d}\n")
                    f.write(f"Length: {chunk['length']} characters\n")
                    f.write(f"Method: {chunk['method']}\n")
                    f.write(f"Section: {chunk['section_id']} - {chunk['section_title']}\n")
                    f.write(f"Section Level: {chunk['section_level']}\n")
                    f.write(f"Pages: {chunk['start_page']}-{chunk['end_page']}\n")
                    f.write(f"Chunk {chunk['chunk_index']} of {chunk['total_chunks']} in this section\n")
                    f.write("-" * 40 + "\n")
                    f.write(chunk['text'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            print(f"✓ Saved detailed chunks TXT to: {chunks_txt_file}")
            print(f"  (Human-readable format)")
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70)
        print(f"Total time: {total_time:.2f}s")
        print(f"Sections processed: {len(sections)}")
        print(f"Chunks created: {total_chunks}")
        if total_chunks > 0:
            print(f"Processing speed: {total_chunks/total_time:.1f} chunks/sec")
        print(f"\n✓ All embeddings stored in ChromaDB")
        print(f"✓ Detailed chunk files saved (without embeddings)")
        print(f"✓ Ready for fast semantic search!")
    
    def _add_text_to_sections(self, sections: List[Dict], pages_data: Dict) -> List[Dict]:
        """
        Recursively add text content to sections from pages data
        
        Args:
            sections: List of section dictionaries
            pages_data: Dictionary mapping page numbers to text
            
        Returns:
            Sections with text content added
        """
        for section in sections:
            # Get text from pages
            section_text = ""
            for page_num in range(section['start_page'], section['end_page'] + 1):
                page_key = str(page_num)
                if page_key in pages_data:
                    section_text += pages_data[page_key] + "\n"
            
            section['text'] = section_text
            
            # Process subsections recursively
            if 'subsections' in section and section['subsections']:
                section['subsections'] = self._add_text_to_sections(
                    section['subsections'], 
                    pages_data
                )
        
        return sections


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: Chunk and store in vector DB simultaneously"
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
        help="Database path (default: ./chroma_db)"
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
    parser.add_argument(
        "--no-chunk-files",
        action="store_true",
        help="Don't save detailed chunk JSON/TXT files"
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = UnifiedPipeline(
        collection_name=args.collection,
        persist_directory=args.db_path,
        buffer_size=args.buffer_size,
        threshold=args.threshold
    )
    
    pipeline.process_sections_file(
        args.sections_file,
        save_metadata=True,
        save_chunks=not args.no_chunk_files
    )


if __name__ == "__main__":
    main()
