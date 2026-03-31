#!/usr/bin/env python3
"""
Main pipeline: JSON processing + chunking comparison
Processes JSON files and runs chunking comparison
"""

import os
import sys
from pathlib import Path

def check_json_files():
    """Check if JSON files are available"""
    # Look for JSON files in input directory
    input_dir = "/app/input" if os.path.exists("/app/input") else "."
    
    # Find JSON files
    json_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.json'):
            json_files.append(file)
    
    if not json_files:
        print("No JSON files found!")
        return False, []
    
    return True, json_files

def run_json_processing():
    """Run JSON text processing"""
    print("Running JSON text processing...")
    
    try:
        # Import and run the JSON processor
        sys.path.append('/app')
        from json_text_processor import main as json_main
        json_main()
        print("JSON processing completed")
        return True
    except Exception as e:
        print(f"JSON processing failed: {e}")
        return False

def run_chunking_comparison():
    """Run chunking comparison"""
    print("Running chunking comparison...")
    
    try:
        # Import and run the chunking comparison
        sys.path.append('/app')
        from chunking_comparison import main as chunking_main
        chunking_main()
        print("Chunking comparison completed")
        return True
    except Exception as e:
        print(f"Chunking comparison failed: {e}")
        return False

def main():
    print("Starting JSON Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Check what we have
    has_json, json_files = check_json_files()
    
    if not has_json:
        print("Please add JSON files to the input directory")
        return
    
    print(f"Found JSON files: {', '.join(json_files)}")
    
    # Step 2: Process JSON files
    if not run_json_processing():
        print("Pipeline failed at JSON processing step")
        return
    
    # Step 3: Check if we have processed sections for comparison
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    section_files = [f for f in os.listdir(output_dir) if f.endswith('_sections.json')]
    
    if not section_files:
        print("No section files found for comparison")
        return
    
    print(f"Found section files: {', '.join(section_files)}")
    
    # Step 4: Run chunking comparison (if available)
    if not run_chunking_comparison():
        print("Chunking comparison failed, but JSON processing completed")
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nGenerated Files:")
    
    # List output files
    for file in os.listdir(output_dir):
        if file.endswith('_sections.txt') or file.endswith('_sections.json') or file.endswith('_chunks_with_pages.txt') or file.endswith('_chunks_with_pages.json'):
            print(f"    {file}")
    
    # List sample files (they should be in /app/samples or current directory)
    sample_dir = "/app/samples" if os.path.exists("/app/samples") else "."
    sample_files = [f for f in os.listdir(sample_dir) if f.startswith('sample_chunks_')]
    
    if sample_files:
        print("\nSample Chunk Files:")
        for file in sample_files:
            print(f"    {file}")
    
    print("\nNext Steps:")
    print("   1. Review the JSON chunking results in *_chunks.txt files")
    print("   2. Check *_chunks.json files for programmatic access")
    print("   3. Review chunking comparison results (if available)")
    print("   4. Examine sample chunk files for quality assessment")

if __name__ == "__main__":
    main()