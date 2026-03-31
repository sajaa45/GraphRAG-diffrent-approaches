#!/usr/bin/env python3
"""
Script to update existing knowledge graph with PREVIOUS_CHUNK relationships
Run this if you already have a knowledge graph and want to add the new relationships
"""

import os
from neo4j import GraphDatabase

def update_knowledge_graph():
    """Add PREVIOUS_CHUNK relationships to existing knowledge graph"""
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Lexical12345")
    
    print("Updating Knowledge Graph with PREVIOUS_CHUNK relationships")
    print("=" * 60)
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Check current state
            result = session.run("MATCH ()-[r:PREVIOUS_CHUNK]->() RETURN count(r) as count")
            existing_prev_rels = result.single()['count']
            
            result = session.run("MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) as count")
            existing_next_rels = result.single()['count']
            
            print(f"Current state:")
            print(f"  NEXT_CHUNK relationships: {existing_next_rels}")
            print(f"  PREVIOUS_CHUNK relationships: {existing_prev_rels}")
            
            if existing_prev_rels > 0:
                print("\nPREVIOUS_CHUNK relationships already exist!")
                response = input("Do you want to recreate them? (y/N): ").lower().strip()
                if response != 'y':
                    print("Skipping update")
                    return
                
                # Delete existing PREVIOUS_CHUNK relationships
                print("Deleting existing PREVIOUS_CHUNK relationships...")
                session.run("MATCH ()-[r:PREVIOUS_CHUNK]->() DELETE r")
            
            # Create PREVIOUS_CHUNK relationships
            print("Creating PREVIOUS_CHUNK relationships...")
            result = session.run("""
                MATCH (s:Section)-[:CONTAINS_CHUNK]->(c1:Chunk)
                MATCH (s)-[:CONTAINS_CHUNK]->(c2:Chunk)
                WHERE c2.chunk_index_in_section = c1.chunk_index_in_section - 1
                MERGE (c1)-[:PREVIOUS_CHUNK]->(c2)
                RETURN count(*) as created
            """)
            
            created_count = result.single()['created']
            print(f"Created {created_count} PREVIOUS_CHUNK relationships")
            
            # Verify final state
            result = session.run("MATCH ()-[r:PREVIOUS_CHUNK]->() RETURN count(r) as count")
            final_prev_rels = result.single()['count']
            
            result = session.run("MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) as count")
            final_next_rels = result.single()['count']
            
            print(f"\nFinal state:")
            print(f"  NEXT_CHUNK relationships: {final_next_rels}")
            print(f"  PREVIOUS_CHUNK relationships: {final_prev_rels}")
            
            if final_next_rels == final_prev_rels:
                print("✅ Success! NEXT_CHUNK and PREVIOUS_CHUNK counts match")
            else:
                print("⚠️  Warning: Relationship counts don't match")
        
        driver.close()
        print("\nKnowledge graph updated successfully!")
        
    except Exception as e:
        print(f"Error updating knowledge graph: {e}")

def main():
    print("Knowledge Graph Updater")
    print("=" * 40)
    
    # Set default connections if not set
    if not os.getenv("NEO4J_URI"):
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    if not os.getenv("NEO4J_USERNAME"):
        os.environ["NEO4J_USERNAME"] = "neo4j"
    if not os.getenv("NEO4J_PASSWORD"):
        print("\nNeo4j password not set. Please set NEO4J_PASSWORD environment variable")
        print("Or run with: NEO4J_PASSWORD=Lexical12345 python update_knowledge_graph.py")
        return
    
    print(f"Neo4j URI: {os.getenv('NEO4J_URI')}")
    print(f"Neo4j Username: {os.getenv('NEO4J_USERNAME')}")
    
    update_knowledge_graph()

if __name__ == "__main__":
    main()