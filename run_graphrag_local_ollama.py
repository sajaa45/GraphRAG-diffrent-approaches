#!/usr/bin/env python3
"""
Script to run GraphRAG with local Ollama (simpler setup)
Assumes you already have Ollama running locally
"""

import subprocess
import requests
import sys

def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return result.returncode == 0

def check_ollama():
    """Check if Ollama is running and has models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                model_names = [model['name'] for model in models]
                print(f"  Ollama is running with models: {model_names}")
                return True
            else:
                print("  Ollama is running but no models found")
                print("Please install a model: ollama pull mistral:latest")
                return False
        else:
            print("  Ollama API returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("  Ollama is not running")
        print("Please start Ollama and install a model:")
        print("  1. Start Ollama")
        print("  2. ollama pull mistral:latest")
        return False
    except Exception as e:
        print(f"  Error checking Ollama: {e}")
        return False

def check_neo4j():
    """Check if Neo4j is accessible"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Lexical12345"))
        with driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) as chunk_count")
            chunk_count = result.single()['chunk_count']
            
            if chunk_count > 0:
                print(f"  Neo4j knowledge graph ready: {chunk_count} chunks")
                driver.close()
                return True
            else:
                print("  Neo4j is running but knowledge graph is empty")
                driver.close()
                return False
    
    except Exception as e:
        print(f"  Neo4j connection failed: {e}")
        print("Please make sure Neo4j is running with the knowledge graph")
        return False

def main():
    print("GraphRAG with Local Ollama")
    print("=" * 40)
    
    # Check prerequisites
    print("\n1. Checking Ollama...")
    if not check_ollama():
        return
    
    print("\n2. Checking Neo4j...")
    if not check_neo4j():
        print("\nStarting Neo4j and building knowledge graph...")
        if not run_command("docker-compose up -d neo4j"):
            print("Failed to start Neo4j")
            return
        
        print("Building knowledge graph...")
        if not run_command("docker-compose up knowledge-graph"):
            print("Failed to build knowledge graph")
            return
    
    # Run GraphRAG with local Ollama
    print("\n3. Running GraphRAG system...")
    print("=" * 40)
    
    # Update docker-compose to use local Ollama
    print("Using local Ollama configuration...")
    
    # Create a temporary override for local Ollama
    override_content = """
services:
  graphrag-test:
    environment:
      - PYTHONUNBUFFERED=1
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=Lexical12345
      - OLLAMA_URL=http://host.docker.internal:11434
    network_mode: "host"
"""
    
    with open("docker-compose.override.yml", "w") as f:
        f.write(override_content)
    
    # Run GraphRAG
    success = run_command("docker-compose up --build graphrag-test", check=False)
    
    # Clean up override file
    run_command("rm -f docker-compose.override.yml", check=False)
    
    if success:
        print("\n  GraphRAG test completed!")
        print("Check graphrag_comparison_results.json for results")
    else:
        print("\n  GraphRAG test failed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")