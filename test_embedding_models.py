#!/usr/bin/env python3
"""
Quick test script to compare different embedding models for chunking
"""
import time
import os
from typing import List, Dict

def test_llamaindex_with_model(text: str, model_name: str) -> Dict:
    """Test LlamaIndex with different embedding models"""
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document
        
        print(f"\nTesting LlamaIndex with {model_name}...")
        
        # Create embedding model
        start_load = time.time()
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        load_time = time.time() - start_load
        
        # Create splitter
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        
        # Process text
        start_process = time.time()
        document = Document(text=text)
        nodes = splitter.get_nodes_from_documents([document])
        process_time = time.time() - start_process
        
        chunks = [node.text for node in nodes]
        
        return {
            "model": model_name,
            "load_time": round(load_time, 2),
            "process_time": round(process_time, 2),
            "total_time": round(load_time + process_time, 2),
            "chunks": len(chunks),
            "avg_length": round(sum(len(c) for c in chunks) / len(chunks), 2) if chunks else 0,
            "success": True
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "error": str(e),
            "success": False
        }

def test_semantic_coherence_models(chunks: List[str]) -> Dict:
    """Test semantic coherence calculation with different models"""
    models_to_test = [
        "all-MiniLM-L6-v2",           # Current (baseline)
        "all-MiniLM-L12-v2",          # Slightly better
        "all-mpnet-base-v2",          # Best balance
        "all-distilroberta-v1"        # Good middle ground
    ]
    
    results = {}
    
    for model_name in models_to_test:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            print(f"\nTesting semantic coherence with {model_name}...")
            
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            # Take small sample for speed
            sample_chunks = chunks[:5] if len(chunks) > 5 else chunks
            
            start_embed = time.time()
            embeddings = model.encode(sample_chunks)
            embed_time = time.time() - start_embed
            
            # Calculate similarities
            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            results[model_name] = {
                "load_time": round(load_time, 2),
                "embed_time": round(embed_time, 2),
                "total_time": round(load_time + embed_time, 2),
                "avg_similarity": round(avg_similarity, 3),
                "success": True
            }
            
        except Exception as e:
            results[model_name] = {
                "error": str(e),
                "success": False
            }
    
    return results

def save_results_to_markdown(llamaindex_results: List[Dict], coherence_results: Dict, output_path: str = None):
    """Save test results to a markdown file"""
    
    if output_path is None:
        output_dir = "/app/output" if os.path.exists("/app/output") else "."
        output_path = os.path.join(output_dir, "embedding_models_test_results.md")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Embedding Models Test Results\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # LlamaIndex Results
            f.write("## LlamaIndex Semantic Chunking Results\n\n")
            f.write("| Model | Load Time (s) | Process Time (s) | Total Time (s) | Chunks | Avg Length | Status |\n")
            f.write("|-------|---------------|------------------|----------------|--------|------------|--------|\n")
            
            for result in llamaindex_results:
                if result["success"]:
                    f.write(f"| {result['model']} | {result['load_time']} | {result['process_time']} | {result['total_time']} | {result['chunks']} | {result['avg_length']} | PASS |\n")
                else:
                    f.write(f"| {result['model']} | - | - | - | - | - | FAIL |\n")
            
            # Performance Analysis
            f.write("\n### Performance Analysis\n\n")
            successful_results = [r for r in llamaindex_results if r["success"]]
            
            if successful_results:
                fastest = min(successful_results, key=lambda x: x["total_time"])
                slowest = max(successful_results, key=lambda x: x["total_time"])
                
                f.write(f"**Fastest Model:** {fastest['model']}\n")
                f.write(f"- Total Time: {fastest['total_time']}s\n")
                f.write(f"- Chunks Created: {fastest['chunks']}\n")
                f.write(f"- Average Chunk Length: {fastest['avg_length']} characters\n\n")
                
                f.write(f"**Slowest Model:** {slowest['model']}\n")
                f.write(f"- Total Time: {slowest['total_time']}s\n")
                f.write(f"- Speed Difference: {slowest['total_time']/fastest['total_time']:.1f}x slower\n\n")
            
            # Semantic Coherence Results
            if coherence_results:
                f.write("## Semantic Coherence Test Results\n\n")
                f.write("| Model | Load Time (s) | Embed Time (s) | Total Time (s) | Avg Similarity | Status |\n")
                f.write("|-------|---------------|----------------|----------------|----------------|--------|\n")
                
                for model, result in coherence_results.items():
                    if result["success"]:
                        f.write(f"| {model} | {result['load_time']} | {result['embed_time']} | {result['total_time']} | {result['avg_similarity']} | PASS |\n")
                    else:
                        f.write(f"| {model} | - | - | - | - | FAIL |\n")
                
                # Coherence Analysis
                f.write("\n### Coherence Analysis\n\n")
                successful_coherence = {k: v for k, v in coherence_results.items() if v["success"]}
                
                if successful_coherence:
                    best_similarity = max(successful_coherence.items(), key=lambda x: x[1]["avg_similarity"])
                    fastest_coherence = min(successful_coherence.items(), key=lambda x: x[1]["total_time"])
                    
                    f.write(f"**Best Semantic Similarity:** {best_similarity[0]}\n")
                    f.write(f"- Average Similarity: {best_similarity[1]['avg_similarity']}\n")
                    f.write(f"- Total Time: {best_similarity[1]['total_time']}s\n\n")
                    
                    f.write(f"**Fastest Coherence Calculation:** {fastest_coherence[0]}\n")
                    f.write(f"- Total Time: {fastest_coherence[1]['total_time']}s\n")
                    f.write(f"- Average Similarity: {fastest_coherence[1]['avg_similarity']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if successful_results:
                # Find best models
                fastest = min(successful_results, key=lambda x: x["total_time"])
                mpnet_result = next((r for r in successful_results if "mpnet" in r["model"]), None)
                l12_result = next((r for r in successful_results if "L12" in r["model"]), None)
                
                f.write("### Speed Priority\n")
                f.write(f"**Recommended:** `{fastest['model']}`\n")
                f.write(f"- Fastest processing at {fastest['total_time']}s\n")
                f.write(f"- Good for real-time applications\n")
                f.write(f"- Creates {fastest['chunks']} chunks\n\n")
                
                if mpnet_result:
                    f.write("### Quality Priority\n")
                    f.write(f"**Recommended:** `{mpnet_result['model']}`\n")
                    f.write(f"- Best semantic understanding\n")
                    f.write(f"- Processing time: {mpnet_result['total_time']}s\n")
                    f.write(f"- Higher quality embeddings\n")
                    f.write(f"- Creates {mpnet_result['chunks']} chunks\n\n")
                
                if l12_result:
                    f.write("### Balanced Option\n")
                    f.write(f"**Recommended:** `{l12_result['model']}`\n")
                    f.write(f"- Good balance of speed and quality\n")
                    f.write(f"- Processing time: {l12_result['total_time']}s\n")
                    f.write(f"- Better than L6, faster than mpnet\n")
                    f.write(f"- Creates {l12_result['chunks']} chunks\n\n")
            
            # Current vs Recommended
            f.write("### Current vs Recommended\n\n")
            f.write("**Current Model:** `sentence-transformers/all-MiniLM-L6-v2`\n")
            f.write("- Good baseline model\n")
            f.write("- Fast and lightweight\n")
            f.write("- Suitable for most applications\n\n")
            
            if successful_results:
                current_result = next((r for r in successful_results if "L6" in r["model"]), None)
                if current_result and mpnet_result:
                    speedup = current_result['total_time'] / mpnet_result['total_time'] if mpnet_result['total_time'] > 0 else 1
                    f.write(f"**Upgrade to mpnet-base-v2:**\n")
                    f.write(f"- Quality improvement: Significant\n")
                    f.write(f"- Speed change: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}\n")
                    f.write(f"- Memory usage: ~3x higher\n")
                    f.write(f"- Recommended for: High-quality document analysis\n\n")
            
            # Implementation Notes
            f.write("## Implementation Notes\n\n")
            f.write("### To Change Model in Code:\n\n")
            f.write("**In `json_text_processor.py` and `chunking_comparison.py`:**\n")
            f.write("```python\n")
            f.write("embed_model = HuggingFaceEmbedding(\n")
            f.write("    model_name=\"sentence-transformers/all-mpnet-base-v2\"  # Change this line\n")
            f.write(")\n")
            f.write("```\n\n")
            
            f.write("**For semantic coherence in `chunking_comparison.py`:**\n")
            f.write("```python\n")
            f.write("model = SentenceTransformer('all-mpnet-base-v2')  # Change this line\n")
            f.write("```\n\n")
            
            # Test Environment
            f.write("## Test Environment\n\n")
            f.write(f"- Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("- Test Text Length: ~10,000 characters\n")
            f.write("- Platform: Docker Container\n")
            f.write("- Python Environment: Isolated container\n\n")
            
            # Errors (if any)
            failed_results = [r for r in llamaindex_results if not r["success"]]
            failed_coherence = {k: v for k, v in coherence_results.items() if not v["success"]} if coherence_results else {}
            
            if failed_results or failed_coherence:
                f.write("## Errors Encountered\n\n")
                
                if failed_results:
                    f.write("### LlamaIndex Failures:\n")
                    for result in failed_results:
                        f.write(f"- **{result['model']}:** {result['error']}\n")
                    f.write("\n")
                
                if failed_coherence:
                    f.write("### Coherence Test Failures:\n")
                    for model, result in failed_coherence.items():
                        f.write(f"- **{model}:** {result['error']}\n")
                    f.write("\n")
        
        print(f"Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

def main():
    # Load some text for testing
    output_dir = "/app/output" if os.path.exists("/app/output") else "output"
    
    # Find JSON section files
    section_files = []
    if os.path.exists(output_dir):
        section_files = [f for f in os.listdir(output_dir) if f.endswith('_sections.json')]
    
    if not section_files:
        print("No JSON section files found. Please run json_text_processor.py first.")
        return
    
    # Load text
    import json
    section_file = os.path.join(output_dir, section_files[0])
    with open(section_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'sections' in data:
        text = "\n\n".join(section['text'] for section in data['sections'])
        # Use smaller sample for testing
        text = text[:10000]  # First 10k chars for speed
    else:
        print("No sections found in JSON file")
        return
    
    print("Testing with {} characters of text".format(len(text)))
    print("="*60)
    
    # Models to test for LlamaIndex
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",     # Current
        "sentence-transformers/all-MiniLM-L12-v2",    # Better version
        "sentence-transformers/all-mpnet-base-v2",    # Best balance
        "sentence-transformers/all-distilroberta-v1", # Good middle
    ]
    
    print("TESTING LLAMAINDEX WITH DIFFERENT MODELS")
    print("="*60)
    
    llamaindex_results = []
    for model in models_to_test:
        result = test_llamaindex_with_model(text, model)
        llamaindex_results.append(result)
        
        if result["success"]:
            print(" {}".format(model))
            print("   Load: {}s, Process: {}s, Total: {}s".format(
                result['load_time'], result['process_time'], result['total_time']))
            print("   Chunks: {}, Avg Length: {}".format(result['chunks'], result['avg_length']))
        else:
            print(" {}: {}".format(model, result['error']))
    
    # Test semantic coherence with different models
    print("\n" + "="*60)
    print("TESTING SEMANTIC COHERENCE MODELS")
    print("="*60)
    
    coherence_results = {}
    
    # Get some chunks first (using current method)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        test_chunks = splitter.split_text(text)
        
        coherence_results = test_semantic_coherence_models(test_chunks)
        
        for model, result in coherence_results.items():
            if result["success"]:
                print(" {}".format(model))
                print("   Load: {}s, Embed: {}s, Total: {}s".format(
                    result['load_time'], result['embed_time'], result['total_time']))
                print("   Avg Similarity: {}".format(result['avg_similarity']))
            else:
                print(" {}: {}".format(model, result['error']))
    
    except ImportError:
        print("LangChain not available for coherence testing")
    
    # Save results to markdown
    print("\n" + "="*60)
    print("SAVING RESULTS TO MARKDOWN")
    print("="*60)
    
    markdown_file = save_results_to_markdown(llamaindex_results, coherence_results)
    if markdown_file:
        print("Results saved successfully!")
    else:
        print("Failed to save results")
    
    # Summary and recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    successful_results = [r for r in llamaindex_results if r["success"]]
    if successful_results:
        # Find fastest
        fastest = min(successful_results, key=lambda x: x["total_time"])
        print("\n FASTEST: {}".format(fastest['model']))
        print("   Time: {}s".format(fastest['total_time']))
        print("   Chunks: {}".format(fastest['chunks']))
        
        # Find best balance (mpnet if available)
        mpnet_result = next((r for r in successful_results if "mpnet" in r["model"]), None)
        if mpnet_result:
            print("\n BEST BALANCE: {}".format(mpnet_result['model']))
            print("   Time: {}s".format(mpnet_result['total_time']))
            print("   Quality: Much better semantic understanding")
    
    print("\n RECOMMENDATION:")
    print("   1. Try 'all-mpnet-base-v2' for best quality/speed balance")
    print("   2. Try 'all-MiniLM-L12-v2' for faster processing with better quality")
    print("   3. Your current model is fine for basic use but these will be much better")
    print("\n Check the generated markdown file for detailed results and implementation notes!")

if __name__ == "__main__":
    main()