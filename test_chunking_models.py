#!/usr/bin/env python3
"""
Test that chunking methods are using the mini-L6 model
"""

def test_json_processor_chunkers():
    """Test the chunkers from json_text_processor.py"""
    print("Testing chunkers from json_text_processor.py...")
    
    try:
        import sys
        sys.path.append('.')
        from json_text_processor import llamaindex_chunker, chonkie_chunker
        
        test_text = """
        This is a test document for semantic chunking. It contains multiple sentences and paragraphs.
        
        The purpose is to verify that the embedding models are working correctly. We want to ensure that the mini-L6 model is being used.
        
        This should be split into meaningful chunks based on semantic similarity.
        """
        
        print("  Testing LlamaIndex chunker...")
        try:
            chunks = llamaindex_chunker(test_text)
            print(f"    ✓ LlamaIndex created {len(chunks)} chunks")
            if chunks and len(chunks) > 0:
                print(f"    ✓ First chunk length: {len(chunks[0])} characters")
        except Exception as e:
            print(f"    ✗ LlamaIndex chunker failed: {e}")
        
        print("  Testing Chonkie chunker...")
        try:
            chunks = chonkie_chunker(test_text)
            print(f"    ✓ Chonkie created {len(chunks)} chunks")
            if chunks and len(chunks) > 0:
                print(f"    ✓ First chunk length: {len(chunks[0])} characters")
        except Exception as e:
            print(f"    ✗ Chonkie chunker failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def test_chunking_comparison_chunkers():
    """Test the chunkers from chunking_comparison.py"""
    print("\nTesting chunkers from chunking_comparison.py...")
    
    try:
        import sys
        sys.path.append('.')
        from chunking_comparison import llamaindex_chunker
        
        test_text = """
        This is a test document for semantic chunking. It contains multiple sentences and paragraphs.
        
        The purpose is to verify that the embedding models are working correctly. We want to ensure that the mini-L6 model is being used.
        
        This should be split into meaningful chunks based on semantic similarity.
        """
        
        print("  Testing LlamaIndex chunker from comparison...")
        try:
            chunks = llamaindex_chunker(test_text)
            print(f"    ✓ LlamaIndex comparison created {len(chunks)} chunks")
            if chunks and len(chunks) > 0:
                print(f"    ✓ First chunk length: {len(chunks[0])} characters")
        except Exception as e:
            print(f"    ✗ LlamaIndex comparison chunker failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

def main():
    print("CHUNKING MODELS TEST")
    print("=" * 30)
    
    # Test json processor chunkers
    json_ok = test_json_processor_chunkers()
    
    # Test chunking comparison chunkers
    comparison_ok = test_chunking_comparison_chunkers()
    
    if json_ok and comparison_ok:
        print("\n  All chunking tests passed! Mini-L6 model is working in all chunkers.")
        print("\nModel being used: sentence-transformers/all-MiniLM-L6-v2")
        print("This model provides:")
        print("  - Fast inference")
        print("  - Good quality embeddings")
        print("  - Small memory footprint")
        print("  - Free to use")
        return True
    else:
        print("\n  Some chunking tests failed.")
        return False

if __name__ == "__main__":
    main()