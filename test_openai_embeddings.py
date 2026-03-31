#!/usr/bin/env python3
"""
Test OpenAI embeddings for comparison (requires API key)
"""

def test_openai_llamaindex(text: str):
    """Test LlamaIndex with OpenAI embeddings"""
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core import Document
        import time
        
        print("Testing LlamaIndex with OpenAI embeddings...")
        
        # Create OpenAI embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",  # Cheaper and faster
            # model="text-embedding-3-large",  # Best quality
        )
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        
        start_time = time.time()
        document = Document(text=text)
        nodes = splitter.get_nodes_from_documents([document])
        end_time = time.time()
        
        chunks = [node.text for node in nodes]
        
        print(f"  OpenAI Embeddings:")
        print(f"   Time: {end_time - start_time:.2f}s")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Avg Length: {sum(len(c) for c in chunks) / len(chunks):.0f}")
        
        return chunks
        
    except ImportError:
        print("  OpenAI embeddings not available. Install with: pip install llama-index-embeddings-openai")
    except Exception as e:
        print(f"  OpenAI error: {e}")
        print("   Make sure OPENAI_API_KEY is set in environment")

if __name__ == "__main__":
    # Load test text
    import json
    import os
    
    output_dir = "output" if os.path.exists("output") else "."
    section_files = [f for f in os.listdir(output_dir) if f.endswith('_sections.json')]
    
    if section_files:
        with open(os.path.join(output_dir, section_files[0]), 'r') as f:
            data = json.load(f)
        text = "\n\n".join(section['text'] for section in data['sections'])[:10000]
        test_openai_llamaindex(text)
    else:
        print("No test data found")