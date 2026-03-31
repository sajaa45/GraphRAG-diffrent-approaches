#!/usr/bin/env python3
"""
Simple test to check if embedding models can be imported
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        print("  Testing sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        print("    ✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"    ✗ sentence-transformers import failed: {e}")
        return False
    
    try:
        print("  Testing llama-index embeddings...")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print("    ✓ llama-index embeddings imported successfully")
    except ImportError as e:
        print(f"    ✗ llama-index embeddings import failed: {e}")
        return False
    
    try:
        print("  Testing llama-index core...")
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.core import Document
        print("    ✓ llama-index core imported successfully")
    except ImportError as e:
        print(f"    ✗ llama-index core import failed: {e}")
        return False
    
    try:
        print("  Testing chonkie...")
        from chonkie import SemanticChunker
        print("    ✓ chonkie imported successfully")
    except ImportError as e:
        print(f"    ✗ chonkie import failed: {e}")
        return False
    
    return True

def test_mini_l6_model():
    """Test if mini-L6 model can be loaded"""
    print("\nTesting mini-L6 model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading all-MiniLM-L6-v2...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("    ✓ Model loaded successfully")
        
        # Test a simple encoding
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        print(f"    ✓ Generated embedding with shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"    ✗ Model loading failed: {e}")
        return False

def main():
    print("SIMPLE EMBEDDING TEST")
    print("=" * 30)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n  Import tests failed. Install missing packages:")
        print("   pip install sentence-transformers")
        print("   pip install llama-index llama-index-embeddings-huggingface")
        print("   pip install chonkie")
        return False
    
    # Test model loading
    model_ok = test_mini_l6_model()
    
    if imports_ok and model_ok:
        print("\n  All tests passed! Ready to use mini-L6 embeddings.")
        return True
    else:
        print("\n  Some tests failed.")
        return False

if __name__ == "__main__":
    main()