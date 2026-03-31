#!/usr/bin/env python3
"""
Test the pipeline locally before Docker
"""

import os
import tempfile
import shutil

def test_pipeline():
    """Test the main pipeline functionality"""
    print("Testing pipeline components...")
    
    # Test 1: Check if JSON processor works
    try:
        from json_text_processor import load_json_pages, should_combine_pages
        print("  JSON processor import successful")
        
        # Test the combination logic
        test_cases = [
            ("This is a test", "Previous sentence.", False),  # Uppercase start, period end
            ("this continues", "Previous sentence", True),    # Lowercase start, no period end
            ("This starts new", "Previous sentence", False),  # Uppercase start, no period end
            ("this continues", "Previous sentence.", False),  # Lowercase start, period end
        ]
        
        for current, previous, expected in test_cases:
            result = should_combine_pages(current, previous)
            if result == expected:
                print(f"  Logic test passed: '{current}' + '{previous}' -> {result}")
            else:
                print(f"  Logic test failed: '{current}' + '{previous}' -> {result}, expected {expected}")
                return False
        
    except ImportError as e:
        print(f"  JSON processor import failed: {e}")
        return False
    
    # Test 2: Check if chunking comparison works
    try:
        from chunking_comparison import langchain_chunker
        test_text = "This is a test. " * 100
        chunks = langchain_chunker(test_text, chunk_size=100, overlap=20)
        if chunks:
            print(f"  Chunking works - created {len(chunks)} chunks")
        else:
            print("  Chunking failed - no chunks created")
            return False
    except ImportError as e:
        print(f"  Chunking import failed: {e}")
        return False
    
    # Test 3: Check if main pipeline imports
    try:
        from main_pipeline import check_json_files
        print("  Main pipeline import successful")
    except ImportError as e:
        print(f"  Main pipeline import failed: {e}")
        return False
    
    print("\n🎉 All pipeline components working!")
    return True

def main():
    print("🧪 Testing JSON Processing Pipeline Components")
    print("=" * 50)
    
    if test_pipeline():
        print("\n  Ready to run JSON processing pipeline!")
        print("Run: docker-compose up --build")
        print("\nMake sure to:")
        print("1. Place your JSON files in the input directory")
        print("2. JSON should have page numbers as keys and text as values")
    else:
        print("\n  Pipeline has issues - check dependencies")

if __name__ == "__main__":
    main()