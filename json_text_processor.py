import json
import os
import re
from typing import List, Dict, Tuple

def load_json_pages(json_path: str) -> Dict[str, str]:
    """
    Load pages from JSON file
    
    Args:
        json_path (str): Path to the JSON file
    
    Returns:
        Dict[str, str]: Dictionary with page numbers as keys and text content as values
    """
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
        
        print(f"Loaded {len(pages_data)} pages from JSON")
        return pages_data
    
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

def should_combine_pages(current_page: str, previous_page: str) -> bool:
    """
    Determine if current page should be combined with previous page based on:
    - Current page doesn't start with uppercase letter
    - Previous page doesn't end with a period
    
    Args:
        current_page (str): Text content of current page
        previous_page (str): Text content of previous page
    
    Returns:
        bool: True if pages should be combined
    """
    if not current_page or not previous_page:
        return False
    
    # Clean and get first non-whitespace character of current page
    current_clean = current_page.strip()
    if not current_clean:
        return False
    
    # Check if current page starts with uppercase letter
    # We need to find the first alphabetic character, not just the first character
    first_alpha_char = None
    for char in current_clean:
        if char.isalpha():
            first_alpha_char = char
            break
    
    # If no alphabetic character found, treat as "starts with uppercase" (don't combine)
    if first_alpha_char is None:
        starts_with_uppercase = True
    else:
        starts_with_uppercase = first_alpha_char.isupper()
    
    # Clean and get last non-whitespace character of previous page
    previous_clean = previous_page.strip()
    if not previous_clean:
        return False
    
    # Check if previous page ends with period
    ends_with_period = previous_clean.endswith('.')
    
    # Combine if current doesn't start with uppercase AND previous doesn't end with period
    should_combine = not starts_with_uppercase and not ends_with_period
    
    return should_combine

def create_page_sections(pages_data: Dict[str, str]) -> List[Dict[str, any]]:
    """
    Create sections from pages according to the specified logic:
    - Each page becomes a section
    - Combine with previous page if current doesn't start with uppercase and previous doesn't end with period
    
    Args:
        pages_data (Dict[str, str]): Dictionary of page numbers and content
    
    Returns:
        List[Dict]: List of sections with metadata
    """
    
    if not pages_data:
        return []
    
    # Sort pages by page number (convert to int for proper sorting)
    sorted_pages = sorted(pages_data.items(), key=lambda x: int(x[0]))
    
    sections = []
    current_section_text = ""
    current_section_pages = []
    
    for i, (page_num, page_text) in enumerate(sorted_pages):
        page_text = page_text.strip()
        
        if i == 0:
            # First page always starts a new section
            current_section_text = page_text
            current_section_pages = [page_num]
        else:
            # Check if we should combine with previous
            previous_page_text = sorted_pages[i-1][1].strip()
            
            if should_combine_pages(page_text, previous_page_text):
                # Combine with current section
                current_section_text += "\n\n" + page_text
                current_section_pages.append(page_num)
                print(f"  Combining page {page_num} with previous section (pages: {current_section_pages})")
            else:
                # Finish current section and start new one
                if current_section_text:
                    sections.append({
                        'text': current_section_text,
                        'pages': current_section_pages.copy(),
                        'page_range': f"{current_section_pages[0]}-{current_section_pages[-1]}" if len(current_section_pages) > 1 else current_section_pages[0],
                        'length': len(current_section_text),
                        'section_id': len(sections) + 1
                    })
                
                # Start new section
                current_section_text = page_text
                current_section_pages = [page_num]
    
    # Don't forget the last section
    if current_section_text:
        sections.append({
            'text': current_section_text,
            'pages': current_section_pages.copy(),
            'page_range': f"{current_section_pages[0]}-{current_section_pages[-1]}" if len(current_section_pages) > 1 else current_section_pages[0],
            'length': len(current_section_text),
            'section_id': len(sections) + 1
        })
    
    return sections

def langchain_chunker(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """LangChain RecursiveCharacterTextSplitter implementation"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        return chunks
    
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return [text]  # Return original text as single chunk

def llamaindex_chunker(text: str) -> List[str]:
    """LlamaIndex SemanticSplitterNodeParser with free embeddings"""
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document
        
        # Use free HuggingFace embeddings instead of OpenAI
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        
        # Create document
        document = Document(text=text)
        nodes = splitter.get_nodes_from_documents([document])
        
        chunks = [node.text for node in nodes]
        return chunks
    
    except ImportError:
        print("LlamaIndex or HuggingFace embeddings not installed.")
        return [text]  # Return original text as single chunk
    except Exception as e:
        print(f"LlamaIndex error: {e}")
        return [text]  # Return original text as single chunk

def chonkie_chunker(text: str) -> List[str]:
    """Chonkie semantic chunker implementation"""
    try:
        from chonkie import SemanticChunker
        
        chunker = SemanticChunker()
        chunks = chunker.chunk(text)
        
        # Extract text from chunk objects
        chunk_texts = [chunk.text for chunk in chunks]
        return chunk_texts
    
    except ImportError:
        print("Chonkie not installed. Install with: pip install chonkie")
        return [text]  # Return original text as single chunk

def apply_chunking_to_sections(sections: List[Dict], method_name: str, **kwargs) -> List[Dict]:
    """
    Apply a chunking method to each section and track page origins
    
    Args:
        sections: List of page sections
        method_name: Name of the chunking method
        **kwargs: Arguments for the chunker function
    
    Returns:
        List of chunks with page tracking information
    """
    
    all_chunks = []
    
    for section in sections:
        print(f"  Processing section {section['section_id']} (pages {section['page_range']}) with {method_name}")
        
        # Apply chunking method to this section
        if method_name == "LangChain":
            section_chunks = langchain_chunker(section['text'], **kwargs)
        elif method_name == "LlamaIndex":
            section_chunks = llamaindex_chunker(section['text'])
        elif method_name == "Chonkie":
            section_chunks = chonkie_chunker(section['text'])
        else:
            section_chunks = [section['text']]  # Fallback: keep as single chunk
        
        # Add metadata to each chunk
        for i, chunk_text in enumerate(section_chunks):
            chunk_info = {
                'text': chunk_text,
                'length': len(chunk_text),
                'section_id': section['section_id'],
                'source_pages': section['pages'],
                'page_range': section['page_range'],
                'chunk_index_in_section': i + 1,
                'total_chunks_in_section': len(section_chunks),
                'method': method_name
            }
            all_chunks.append(chunk_info)
    
    return all_chunks

def save_sections_to_file(sections: List[Dict], output_path: str = None):
    """
    Save page sections to a text file for inspection
    
    Args:
        sections (List[Dict]): List of section dictionaries
        output_path (str, optional): Output file path
    """
    
    if output_path is None:
        output_dir = "/app/output" if os.path.exists("/app/output") else "."
        output_path = os.path.join(output_dir, "page_sections.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"JSON Page Sections Results\n")
            f.write(f"Total sections: {len(sections)}\n")
            f.write("=" * 80 + "\n\n")
            
            for section in sections:
                f.write(f"SECTION {section['section_id']}\n")
                f.write(f"Pages: {section['page_range']}\n")
                f.write(f"Length: {section['length']} characters\n")
                f.write(f"Page numbers: {', '.join(section['pages'])}\n")
                f.write("-" * 40 + "\n")
                f.write(section['text'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"Sections saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving sections: {e}")

def save_method_chunks_to_file(chunks: List[Dict], method_name: str, output_path: str = None):
    """
    Save chunks from a specific method to a text file for inspection
    
    Args:
        chunks (List[Dict]): List of chunk dictionaries
        method_name (str): Name of the chunking method
        output_path (str, optional): Output file path
    """
    
    if output_path is None:
        output_dir = "/app/output" if os.path.exists("/app/output") else "."
        safe_method_name = method_name.lower().replace(' ', '_')
        output_path = os.path.join(output_dir, f"{safe_method_name}_chunks_with_pages.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{method_name} Chunking Results\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks, 1):
                f.write(f"CHUNK {i:04d}\n")
                f.write(f"Method: {chunk['method']}\n")
                f.write(f"Section: {chunk['section_id']}\n")
                f.write(f"Source Pages: {chunk['page_range']}\n")
                f.write(f"Page Numbers: {', '.join(chunk['source_pages'])}\n")
                f.write(f"Chunk {chunk['chunk_index_in_section']} of {chunk['total_chunks_in_section']} in this section\n")
                f.write(f"Length: {chunk['length']} characters\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['text'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"{method_name} chunks saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving {method_name} chunks: {e}")

def analyze_sections(sections: List[Dict]) -> Dict[str, any]:
    """
    Analyze the section results and provide statistics
    
    Args:
        sections (List[Dict]): List of section dictionaries
    
    Returns:
        Dict: Analysis results
    """
    
    if not sections:
        return {"error": "No sections to analyze"}
    
    # Basic statistics
    total_sections = len(sections)
    section_lengths = [section['length'] for section in sections]
    total_characters = sum(section_lengths)
    
    # Page combination statistics
    single_page_sections = sum(1 for section in sections if len(section['pages']) == 1)
    multi_page_sections = sum(1 for section in sections if len(section['pages']) > 1)
    max_pages_in_section = max(len(section['pages']) for section in sections)
    
    # Length statistics
    avg_length = total_characters / total_sections
    min_length = min(section_lengths)
    max_length = max(section_lengths)
    
    analysis = {
        "total_sections": total_sections,
        "total_characters": total_characters,
        "average_section_length": round(avg_length, 2),
        "min_section_length": min_length,
        "max_section_length": max_length,
        "single_page_sections": single_page_sections,
        "multi_page_sections": multi_page_sections,
        "max_pages_per_section": max_pages_in_section,
        "combination_rate": round(multi_page_sections / total_sections * 100, 2)
    }
    
    return analysis

def main():
    """Main function to process JSON and create sections, then apply chunking methods"""
    
    # Configuration - check for Docker environment
    input_dir = "/app/input" if os.path.exists("/app/input") else "."
    output_dir = "/app/output" if os.path.exists("/app/output") else "."
    
    # Look for JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        
        print(f"\nProcessing: {json_file}")
        print("=" * 60)
        
        # Check if output files already exist
        sections_file = os.path.join(output_dir, json_file.replace('.json', '_sections.txt'))
        sections_json_file = os.path.join(output_dir, json_file.replace('.json', '_sections.json'))
        
        if os.path.exists(sections_file) and os.path.exists(sections_json_file):
            print(f"Section files already exist for {json_file}")
            
            # Check if method chunk files also exist
            method_files_exist = all(
                os.path.exists(os.path.join(output_dir, f"{method.lower()}_chunks_with_pages.txt")) and
                os.path.exists(os.path.join(output_dir, f"{method.lower()}_chunks_with_pages.json"))
                for method in ["llamaindex"]
            )
            
            if method_files_exist:
                print(f"All method chunk files already exist for {json_file}")
                print("   Skipping processing (files already exist)")
                continue
            else:
                print("   Some method chunk files missing, will process chunking methods only")
                # Load existing sections
                try:
                    with open(sections_json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sections = data.get('sections', [])
                    print(f"   Loaded {len(sections)} existing sections")
                except Exception as e:
                    print(f"   Error loading existing sections: {e}")
                    continue
        else:
            # Load pages from JSON
            pages_data = load_json_pages(json_path)
            
            if not pages_data:
                print(f"Failed to load data from {json_file}")
                continue
            
            # STEP 1: Create page sections using the specified logic
            print("Step 1: Creating page sections...")
            sections = create_page_sections(pages_data)
            
            if not sections:
                print("No sections created")
                continue
            
            # Analyze section results
            analysis = analyze_sections(sections)
            
            print(f"\nSection Results:")
            print(f"  Total sections: {analysis['total_sections']}")
            print(f"  Single-page sections: {analysis['single_page_sections']}")
            print(f"  Multi-page sections: {analysis['multi_page_sections']}")
            print(f"  Page combination rate: {analysis['combination_rate']}%")
            print(f"  Average section length: {analysis['average_section_length']} characters")
            print(f"  Length range: {analysis['min_section_length']} - {analysis['max_section_length']} characters")
            print(f"  Max pages per section: {analysis['max_pages_per_section']}")
            
            # Save sections to file
            save_sections_to_file(sections, sections_file)
            
            # Save sections as JSON for programmatic use
            try:
                with open(sections_json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'analysis': analysis,
                        'sections': sections
                    }, f, indent=2, ensure_ascii=False)
                print(f"Sections JSON saved to: {sections_json_file}")
            except Exception as e:
                print(f"Error saving sections JSON: {e}")
        
        # STEP 2: Apply chunking methods to each section
        print(f"\nStep 2: Applying chunking methods to {len(sections)} sections...")
        
        chunking_methods = [
            ("LangChain", {"chunk_size": 512, "overlap": 64}),
            ("LlamaIndex", {}),
            ("Chonkie", {})
        ]
        
        for method_name, kwargs in chunking_methods:
            # Check if method files already exist
            method_txt_file = os.path.join(output_dir, f"{method_name.lower()}_chunks_with_pages.txt")
            method_json_file = os.path.join(output_dir, f"{method_name.lower()}_chunks_with_pages.json")
            
            if os.path.exists(method_txt_file) and os.path.exists(method_json_file):
                print(f"{method_name} chunk files already exist, skipping")
                continue
            
            print(f"\n--- Applying {method_name} chunking ---")
            
            try:
                chunks = apply_chunking_to_sections(sections, method_name, **kwargs)
                
                if chunks:
                    print(f"  Created {len(chunks)} chunks total")
                    
                    # Save chunks to file
                    save_method_chunks_to_file(chunks, method_name)
                    
                    # Save chunks as JSON
                    try:
                        with open(method_json_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'method': method_name,
                                'total_chunks': len(chunks),
                                'chunks': chunks
                            }, f, indent=2, ensure_ascii=False)
                        print(f"  {method_name} chunks JSON saved to: {method_json_file}")
                    except Exception as e:
                        print(f"  Error saving {method_name} chunks JSON: {e}")
                else:
                    print(f"  No chunks created for {method_name}")
                    
            except Exception as e:
                print(f"  Error applying {method_name}: {e}")

if __name__ == "__main__":
    main()