#!/usr/bin/env python3
"""
GraphRAG System using Neo4j Knowledge Graph and Local Ollama
Tests different retrieval approaches for query answering
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import requests

class GraphRAGSystem:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, 
                 ollama_url: str = "http://localhost:11434", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize GraphRAG system
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            ollama_url: Ollama API URL
            embedding_model: Sentence transformer model for embeddings
        """
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        self.ollama_url = ollama_url
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Test connections
        self._test_neo4j_connection()
        self._test_ollama_connection()
        
        print(f"GraphRAG System initialized successfully")
        print(f"- Neo4j: {neo4j_uri}")
        print(f"- Ollama: {ollama_url}")
        print(f"- Embedding model: {embedding_model}")
    
    def _test_neo4j_connection(self):
        """Test Neo4j connection"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN count(*) as total_nodes")
                total = result.single()['total_nodes']
                print(f"Neo4j connected: {total} total nodes")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def _test_ollama_connection(self):
        """Test Ollama connection and list available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                print(f"Ollama connected: {len(models)} models available")
                print(f"Available models: {model_names}")
                return model_names
            else:
                raise ConnectionError(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to Ollama. Make sure Ollama is running on localhost:11434")
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def embed_query(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        return self.embedding_model.encode([query])[0]
    
    def find_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Find most similar chunks using Neo4j and cosine similarity
        Note: This is a simplified approach. In production, you'd store embeddings in Neo4j
        """
        with self.neo4j_driver.session() as session:
            # Get all chunks with their content
            result = session.run("""
                MATCH (c:Chunk)
                RETURN c.chunk_id as chunk_id, 
                       c.content as content,
                       c.section_id as section_id,
                       c.page_range as page_range,
                       c.chunk_index_in_section as chunk_index,
                       c.total_chunks_in_section as total_chunks
                ORDER BY c.chunk_id
            """)
            
            chunks = []
            chunk_texts = []
            
            for record in result:
                chunk_data = {
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'section_id': record['section_id'],
                    'page_range': record['page_range'],
                    'chunk_index': record['chunk_index'],
                    'total_chunks': record['total_chunks']
                }
                chunks.append(chunk_data)
                chunk_texts.append(record['content'])
            
            print(f"Loaded {len(chunks)} chunks for similarity search")
            
            # Calculate embeddings for all chunks (in production, store these in Neo4j)
            print("Calculating chunk embeddings...")
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            # Calculate cosine similarities
            similarities = np.dot(chunk_embeddings, query_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_chunks = []
            for idx in top_indices:
                chunk = chunks[idx].copy()
                chunk['similarity'] = float(similarities[idx])
                similar_chunks.append(chunk)
            
            return similar_chunks
    
    def get_next_chunk(self, chunk_id: str) -> Dict:
        """Get the next chunk in sequence using Neo4j relationships"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c1:Chunk {chunk_id: $chunk_id})-[:NEXT_CHUNK]->(c2:Chunk)
                RETURN c2.chunk_id as chunk_id,
                       c2.content as content,
                       c2.section_id as section_id,
                       c2.page_range as page_range,
                       c2.chunk_index_in_section as chunk_index
            """, chunk_id=chunk_id)
            
            record = result.single()
            if record:
                return {
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'section_id': record['section_id'],
                    'page_range': record['page_range'],
                    'chunk_index': record['chunk_index']
                }
            return None
    
    def get_previous_chunk(self, chunk_id: str) -> Dict:
        """Get the previous chunk in sequence using Neo4j relationships"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c1:Chunk {chunk_id: $chunk_id})-[:PREVIOUS_CHUNK]->(c2:Chunk)
                RETURN c2.chunk_id as chunk_id,
                       c2.content as content,
                       c2.section_id as section_id,
                       c2.page_range as page_range,
                       c2.chunk_index_in_section as chunk_index
            """, chunk_id=chunk_id)
            
            record = result.single()
            if record:
                return {
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'section_id': record['section_id'],
                    'page_range': record['page_range'],
                    'chunk_index': record['chunk_index']
                }
            return None
    
    def get_section_chunks(self, section_id: str) -> List[Dict]:
        """Get all chunks from the same section"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                RETURN c.chunk_id as chunk_id,
                       c.content as content,
                       c.chunk_index_in_section as chunk_index,
                       c.page_range as page_range
                ORDER BY c.chunk_index_in_section
            """, section_id=section_id)
            
            chunks = []
            for record in result:
                chunks.append({
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'chunk_index': record['chunk_index'],
                    'page_range': record['page_range']
                })
            
            return chunks
    
    def query_ollama(self, prompt: str, model: str = "mistral:latest", max_tokens: int = 500) -> Dict:
        """Query Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1
                }
            }
            
            start_time = time.time()
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "model": model,
                    "response_time": round(end_time - start_time, 2),
                    "success": True
                }
            else:
                return {
                    "error": f"Ollama API error: {response.status_code}",
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to query Ollama: {e}",
                "success": False
            }
    
    def approach_1_single_chunk(self, query: str, model: str = "mistral:latest") -> Dict:
        """Approach 1: Use only the most similar chunk"""
        print(f"\n=== Approach 1: Single Chunk Retrieval ===")
        
        # Find most similar chunk
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        print(f"Pages: {best_chunk['page_range']}")
        
        # Create prompt
        prompt = f"""Based on the following document excerpt, please answer the question.

Document excerpt:
{best_chunk['content']}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "single_chunk"
        result['chunks_used'] = 1
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = best_chunk['page_range']
        result['chunk_ids'] = [best_chunk['chunk_id']]
        
        return result
    
    def approach_2_sequential_chunks(self, query: str, model: str = "mistral:latest") -> Dict:
        """Approach 2: Use the most similar chunk + next chunk"""
        print(f"\n=== Approach 2: Sequential Chunks Retrieval ===")
        
        # Find most similar chunk
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        
        # Get next chunk
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = [best_chunk]
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found")
        
        # Combine content
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, please answer the question.

Document excerpts:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "sequential_chunks"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        
        return result
    

    
    def approach_4_context_window(self, query: str, model: str = "mistral:latest") -> Dict:
        """Approach 4: Use previous + current + next chunk (context window)"""
        print(f"\n=== Approach 4: Context Window Retrieval (Prev + Current + Next) ===")
        
        # Find most similar chunk
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        
        # Get previous and next chunks
        previous_chunk = self.get_previous_chunk(best_chunk['chunk_id'])
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = []
        
        # Add previous chunk if available
        if previous_chunk:
            chunks_to_use.append(previous_chunk)
            print(f"Previous chunk: {previous_chunk['chunk_id']}")
        else:
            print("No previous chunk found")
        
        # Add the best matching chunk
        chunks_to_use.append(best_chunk)
        
        # Add next chunk if available
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found")
        
        print(f"Using {len(chunks_to_use)} chunks in context window")
        
        # Combine content in order
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpts (in sequential order), please answer the question.

Document excerpts:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "context_window"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        result['has_previous'] = previous_chunk is not None
        result['has_next'] = next_chunk is not None
        
        return result
    
    def approach_1_single_chunk_optimized(self, query: str, model: str, best_chunk: Dict) -> Dict:
        """Optimized Approach 1: Use only the most similar chunk (pre-calculated)"""
        print(f"\n=== Approach 1: Single Chunk Retrieval ===")
        print(f"Using pre-found chunk: {best_chunk['chunk_id']}")
        
        # Create prompt
        prompt = f"""Based on the following document excerpt, please answer the question.

Document excerpt:
{best_chunk['content']}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "single_chunk"
        result['chunks_used'] = 1
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = best_chunk['page_range']
        result['chunk_ids'] = [best_chunk['chunk_id']]
        
        return result
    
    def approach_2_sequential_chunks_optimized(self, query: str, model: str, best_chunk: Dict) -> Dict:
        """Optimized Approach 2: Use the most similar chunk + next chunk (pre-calculated)"""
        print(f"\n=== Approach 2: Sequential Chunks Retrieval ===")
        print(f"Using pre-found chunk: {best_chunk['chunk_id']}")
        
        # Get next chunk
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = [best_chunk]
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found - using only the best chunk")
        
        # Combine content
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpt{"s" if len(chunks_to_use) > 1 else ""}, please answer the question.

Document excerpt{"s" if len(chunks_to_use) > 1 else ""}:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "sequential_chunks"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        result['has_next'] = next_chunk is not None
        
        return result
    
    def approach_4_context_window_optimized(self, query: str, model: str, best_chunk: Dict) -> Dict:
        """Optimized Approach 3: Use previous + current + next chunk (pre-calculated)"""
        print(f"\n=== Approach 3: Context Window Retrieval (Prev + Current + Next) ===")
        print(f"Using pre-found chunk: {best_chunk['chunk_id']}")
        
        # Get previous and next chunks
        previous_chunk = self.get_previous_chunk(best_chunk['chunk_id'])
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = []
        
        # Add previous chunk if available
        if previous_chunk:
            chunks_to_use.append(previous_chunk)
            print(f"Previous chunk: {previous_chunk['chunk_id']}")
        else:
            print("No previous chunk found")
        
        # Add the best matching chunk
        chunks_to_use.append(best_chunk)
        
        # Add next chunk if available
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found")
        
        print(f"Using {len(chunks_to_use)} chunks in context window")
        
        # Combine content in order
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpt{"s" if len(chunks_to_use) > 1 else ""} (in sequential order), please answer the question.

Document excerpt{"s" if len(chunks_to_use) > 1 else ""}:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "context_window"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        result['has_previous'] = previous_chunk is not None
        result['has_next'] = next_chunk is not None
        
        return result
    
    def compare_approaches(self, query: str, model: str = "mistral:latest") -> Dict:
        """Compare all three approaches for a given query"""
        print(f"\n{'='*80}")
        print(f"COMPARING GRAPHRAG APPROACHES")
        print(f"Query: {query}")
        print(f"Model: {model}")
        print(f"{'='*80}")
        
        # Calculate embeddings once for efficiency
        print("Finding most similar chunk (calculating embeddings once)...")
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found for any approach", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        print(f"Pages: {best_chunk['page_range']}")
        
        results = {}
        
        # Test each approach using the pre-found best chunk
        results['approach_1'] = self.approach_1_single_chunk_optimized(query, model, best_chunk)
        results['approach_2'] = self.approach_2_sequential_chunks_optimized(query, model, best_chunk)
        results['approach_3'] = self.approach_4_context_window_optimized(query, model, best_chunk)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for approach_name, result in results.items():
            if result['success']:
                print(f"\n{approach_name.upper().replace('_', ' ')}:")
                print(f"  Chunks used: {result['chunks_used']}")
                print(f"  Response time: {result['response_time']}s")
                print(f"  Similarity score: {result.get('similarity_score', 'N/A'):.3f}")
                print(f"  Source pages: {result['source_pages']}")
                if approach_name == 'approach_3':
                    print(f"  Has previous: {result.get('has_previous', False)}")
                    print(f"  Has next: {result.get('has_next', False)}")
                print(f"  Response preview: {result['response'][:200]}...")
            else:
                print(f"\n{approach_name.upper().replace('_', ' ')}: FAILED")
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        return results
    
    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

def main():
    """Main function to test GraphRAG system"""
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Lexical12345")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
    
    print("GraphRAG System with Neo4j Knowledge Graph and Ollama")
    print("=" * 60)
    
    try:
        # Initialize system
        graphrag = GraphRAGSystem(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OLLAMA_URL)
        
        # Get available models
        models = graphrag.get_available_ollama_models()
        if not models:
            print("No Ollama models found. Please install a model first:")
            print(f"  ollama pull {OLLAMA_MODEL}")
            return
        
        # Use specified model or first available
        model_to_use = OLLAMA_MODEL if OLLAMA_MODEL in models else models[0]
        print(f"Using model: {model_to_use}")
        
        if model_to_use != OLLAMA_MODEL:
            print(f"Note: Requested model '{OLLAMA_MODEL}' not found, using '{model_to_use}' instead")
        
        # Test queries
        test_queries = [
            "Who is the Chairman of the Board, and what is his medical background?",
            "What is the estimated Gross Development Value (GDV) for the future development at Genting Highlands?",
            "which year did the company report its highest Revenue?",
            "What was the Earnings Per Share (EPS) in 2019?",
        ]
        
        # Test each query with all approaches
        all_results = {}
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n\n{'#'*100}")
            print(f"TEST QUERY {i}: {query}")
            print(f"{'#'*100}")
            
            results = graphrag.compare_approaches(query, model_to_use)
            all_results[f"query_{i}"] = {
                "query": query,
                "results": results
            }
        
        # Save results to file
        output_file = "/app/output/graphrag_comparison_results.json" if os.path.exists("/app/output") else "graphrag_comparison_results.json"
        
        # Add metadata to results
        final_results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": model_to_use,
                "neo4j_uri": NEO4J_URI,
                "ollama_url": OLLAMA_URL,
                "total_queries": len(test_queries),
                "approaches": [
                    {"id": "approach_1", "name": "Single Chunk", "description": "Uses only the most similar chunk"},
                    {"id": "approach_2", "name": "Sequential Chunks", "description": "Uses the most similar chunk + next chunk"},
                    {"id": "approach_3", "name": "Context Window", "description": "Uses previous + current + next chunk"}
                ]
            },
            "queries": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n\nResults saved to: {output_file}")
        
        # Performance summary
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        approach_stats = {
            'approach_1': {'total_time': 0, 'success_count': 0},
            'approach_2': {'total_time': 0, 'success_count': 0},
            'approach_3': {'total_time': 0, 'success_count': 0}
        }
        
        for query_data in all_results.values():
            for approach, result in query_data['results'].items():
                if result['success']:
                    approach_stats[approach]['total_time'] += result['response_time']
                    approach_stats[approach]['success_count'] += 1
        
        for approach, stats in approach_stats.items():
            if stats['success_count'] > 0:
                avg_time = stats['total_time'] / stats['success_count']
                approach_name = "Context Window" if approach == "approach_3" else approach.replace('_', ' ').title()
                print(f"{approach_name}: {avg_time:.2f}s average, {stats['success_count']}/{len(test_queries)} successful")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'graphrag' in locals():
            graphrag.close()

if __name__ == "__main__":
    main()