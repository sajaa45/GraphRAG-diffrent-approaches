#!/usr/bin/env python3
"""
Test embedding models for chunking
"""

import time
import os
from typing import List

def test_sentence_transformers():
    """Test sentence-transformers mini-L6 model"""
    print("Testing sentence-transformers all-MiniLM-L6-v2...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the model
        print("  Loading model...")
        start_time = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        print(f"  Model loaded in {load_time:.2f} seconds")
        
        # Test with sample text
        test_texts = [
            "This is a test sentence about artificial intelligence.",
            "Machine learning models are becoming more sophisticated.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        print("  Generating embeddings...")
        start_time = time.time()
        embeddings = model.encode(test_texts)
        embed_time = time.time() - start_time
        
        print(f"  Generated {len(embeddings)} embeddings in {embed_time:.2f} seconds")
        print(f"  Embedding dimension: {embeddings[0].shape}")
        print(f"  Average embedding time: {embed_time/len(test_texts):.3f} seconds per text")
        
        # Test similarity
        import numpy as np
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"  Sample similarity between first two texts: {similarity:.3f}")
        
        return True, model
        
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Install with: pip install sentence-transformers")
        return False, None
    except Exception as e:
        print(f"  Error: {e}")
        return False, None

def test_huggingface_embeddings():
    """Test HuggingFace embeddings for LlamaIndex"""
    print("\nTesting HuggingFace embeddings for LlamaIndex...")
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print("  Loading HuggingFace embedding model...")
        start_time = time.time()
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        load_time = time.time() - start_time
        print(f"  Model loaded in {load_time:.2f} seconds")
        
        # Test embedding
        test_text = "This is a test document for semantic chunking with LlamaIndex."
        
        print("  Generating embedding...")
        start_time = time.time()
        embedding = embed_model.get_text_embedding(test_text)
        embed_time = time.time() - start_time
        
        print(f"  Generated embedding in {embed_time:.2f} seconds")
        print(f"  Embedding dimension: {len(embedding)}")
        
        return True, embed_model
        
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Install with: pip install llama-index-embeddings-huggingface")
        return False, None
    except Exception as e:
        print(f"  Error: {e}")
        return False, None

def test_llamaindex_semantic_chunker():
    """Test LlamaIndex semantic chunker with mini-L6"""
    print("\nTesting LlamaIndex SemanticSplitterNodeParser...")
    
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document
        
        # Sample text for testing
        test_text = """
        Artificial intelligence is transforming the world. Machine learning algorithms can now process vast amounts of data.
        
        Natural language processing enables computers to understand human language. This technology powers chatbots and translation services.
        
        Deep learning uses neural networks with multiple layers. These networks can recognize patterns in images, text, and speech.
        
        The future of AI looks promising. New breakthroughs are happening every day in research labs around the world.
        """
        
        print("  Setting up semantic splitter...")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        
        print("  Creating document and chunking...")
        start_time = time.time()
        document = Document(text=test_text)
        nodes = splitter.get_nodes_from_documents([document])
        chunk_time = time.time() - start_time
        
        chunks = [node.text for node in nodes]
        
        print(f"  Created {len(chunks)} chunks in {chunk_time:.2f} seconds")
        print("  Chunk lengths:", [len(chunk) for chunk in chunks])
        
        # Show first chunk as example
        if chunks:
            print(f"  First chunk preview: {chunks[0][:100]}...")
        
        return True, chunks
        
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Install with: pip install llama-index llama-index-embeddings-huggingface")
        return False, None
    except Exception as e:
        print(f"  Error: {e}")
        return False, None

def test_chonkie_chunker():
    """Test Chonkie semantic chunker"""
    print("\nTesting Chonkie SemanticChunker...")
    
    try:
        from chonkie import SemanticChunker
        
        test_text = """
        Artificial intelligence is transforming the world. Machine learning algorithms can now process vast amounts of data.
        
        Natural language processing enables computers to understand human language. This technology powers chatbots and translation services.
        
        Deep learning uses neural networks with multiple layers. These networks can recognize patterns in images, text, and speech.
        
        The future of AI looks promising. New breakthroughs are happening every day in research labs around the world.
        """
        
        print("  Creating Chonkie chunker...")
        start_time = time.time()
        chunker = SemanticChunker()
        setup_time = time.time() - start_time
        print(f"  Chunker setup in {setup_time:.2f} seconds")
        
        print("  Chunking text...")
        start_time = time.time()
        chunks = chunker.chunk(test_text)
        chunk_time = time.time() - start_time
        
        chunk_texts = [chunk.text for chunk in chunks]
        
        print(f"  Created {len(chunk_texts)} chunks in {chunk_time:.2f} seconds")
        print("  Chunk lengths:", [len(chunk) for chunk in chunk_texts])
        
        # Show first chunk as example
        if chunk_texts:
            print(f"  First chunk preview: {chunk_texts[0][:100]}...")
        
        return True, chunk_texts
        
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Install with: pip install chonkie")
        return False, None
    except Exception as e:
        print(f"  Error: {e}")
        return False, None

def main():
    print("EMBEDDING MODELS TEST")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Sentence Transformers
    success, model = test_sentence_transformers()
    results['sentence_transformers'] = success
    
    # Test 2: HuggingFace Embeddings for LlamaIndex
    success, embed_model = test_huggingface_embeddings()
    results['huggingface_embeddings'] = success
    
    # Test 3: LlamaIndex Semantic Chunker
    success, chunks = test_llamaindex_semantic_chunker()
    results['llamaindex_chunker'] = success
    
    # Test 4: Chonkie Chunker
    success, chunks = test_chonkie_chunker()
    results['chonkie_chunker'] = success
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<25} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\nAll tests passed! Embedding models are working correctly.")
    else:
        print("\nSome tests failed. Check the error messages above.")
        failed_tests = [name for name, success in results.items() if not success]
        print(f"Failed tests: {', '.join(failed_tests)}")
    
    return all_passed

if __name__ == "__main__":
    main()