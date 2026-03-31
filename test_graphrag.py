#!/usr/bin/env python3
"""
Simple script to test GraphRAG system locally
Make sure you have:
1. Neo4j running with knowledge graph
2. Ollama running with a model installed
"""

import os
import sys

def main():
    print("GraphRAG System Tester")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        "output/llamaindex_chunks_with_pages.json",
        "output/pdf2_sections.json", 
        "pdf2.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run the knowledge graph builder first")
        return
    
    print("All required files found")
    
    # Set default connections if not set
    if not os.getenv("NEO4J_URI"):
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    if not os.getenv("NEO4J_USERNAME"):
        os.environ["NEO4J_USERNAME"] = "neo4j"
    if not os.getenv("NEO4J_PASSWORD"):
        print("\nNeo4j password not set. Please set NEO4J_PASSWORD environment variable")
        print("Or run with: NEO4J_PASSWORD=Lexical12345 python test_graphrag.py")
        return
    
    if not os.getenv("OLLAMA_URL"):
        os.environ["OLLAMA_URL"] = "http://localhost:11434"
    
    print(f"Neo4j URI: {os.getenv('NEO4J_URI')}")
    print(f"Neo4j Username: {os.getenv('NEO4J_USERNAME')}")
    print(f"Ollama URL: {os.getenv('OLLAMA_URL')}")
    
    # Import and run the GraphRAG system
    try:
        from graphrag_system import main as run_graphrag
        run_graphrag()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()