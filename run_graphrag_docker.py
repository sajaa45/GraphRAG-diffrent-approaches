#!/usr/bin/env python3
"""
Script to run GraphRAG system with full Docker stack
Handles Ollama model setup and service orchestration
"""

import subprocess
import time
import sys
import requests

def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return result.returncode == 0

def wait_for_service(url, service_name, max_wait=60):
    """Wait for a service to be ready"""
    print(f"Waiting for {service_name} to be ready...")
    for i in range(max_wait):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"{service_name} is ready!")
                return True
        except:
            pass
        time.sleep(1)
        if i % 10 == 0:
            print(f"Still waiting for {service_name}... ({i}s)")
    
    print(f"Timeout waiting for {service_name}")
    return False

def main():
    print("GraphRAG Docker Setup and Runner")
    print("=" * 50)
    
    # Step 1: Start the services
    print("\n1. Starting Docker services...")
    if not run_command("docker-compose up -d neo4j ollama"):
        print("Failed to start services")
        return
    
    # Step 2: Wait for Ollama to be ready
    print("\n2. Waiting for Ollama to start...")
    if not wait_for_service("http://localhost:11434/api/tags", "Ollama", 120):
        print("Ollama failed to start")
        return
    
    # Step 3: Check if models are available
    print("\n3. Checking available models...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        if not models:
            print("No models found. Installing mistral:latest...")
            print("This may take several minutes...")
            if not run_command("docker exec ollama-server ollama pull mistral:latest"):
                print("Failed to pull model")
                return
        else:
            print(f"Found models: {model_names}")
    
    except Exception as e:
        print(f"Error checking models: {e}")
        return
    
    # Step 4: Wait for Neo4j (if not already running)
    print("\n4. Waiting for Neo4j...")
    if not wait_for_service("http://localhost:7474", "Neo4j", 60):
        print("Neo4j failed to start")
        return
    
    # Step 5: Check if knowledge graph exists
    print("\n5. Checking knowledge graph...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Lexical12345"))
        with driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) as chunk_count")
            chunk_count = result.single()['chunk_count']
            
            if chunk_count == 0:
                print("No chunks found in knowledge graph. Building knowledge graph first...")
                if not run_command("docker-compose up knowledge-graph"):
                    print("Failed to build knowledge graph")
                    return
            else:
                print(f"Knowledge graph ready: {chunk_count} chunks found")
        
        driver.close()
    
    except Exception as e:
        print(f"Error checking knowledge graph: {e}")
        print("Building knowledge graph...")
        if not run_command("docker-compose up knowledge-graph"):
            print("Failed to build knowledge graph")
            return
    
    # Step 6: Run GraphRAG system
    print("\n6. Running GraphRAG system...")
    print("=" * 50)
    
    # Run in foreground to see output
    run_command("docker-compose up graphrag-test", check=False)
    
    print("\nGraphRAG test completed!")
    print("Check graphrag_comparison_results.json for detailed results")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping services...")
        run_command("docker-compose down", check=False)
    except Exception as e:
        print(f"Error: {e}")
        run_command("docker-compose down", check=False)