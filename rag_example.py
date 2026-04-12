#!/usr/bin/env python3
"""
Example RAG (Retrieval Augmented Generation) pipeline
Uses the vector store for fast, hierarchical retrieval
"""

from typing import List, Dict
from query_vector_store import VectorStoreQuery


class RAGPipeline:
    """Simple RAG pipeline using hierarchical vector store"""
    
    def __init__(self, 
                 collection_name: str = "financial_docs",
                 persist_directory: str = "./chroma_db"):
        """Initialize RAG pipeline"""
        print("Initializing RAG pipeline...")
        self.query_interface = VectorStoreQuery(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        print("✓ Ready")
    
    def retrieve_context(self, 
                        query: str,
                        n_sections: int = 2,
                        n_chunks_per_section: int = 3) -> Dict:
        """
        Retrieve relevant context for a query
        
        Returns:
            Dictionary with sections and chunks
        """
        results = self.query_interface.hierarchical_query(
            query,
            n_sections=n_sections,
            n_chunks_per_section=n_chunks_per_section
        )
        return results
    
    def format_context(self, results: Dict) -> str:
        """
        Format retrieved results into context string
        
        Args:
            results: Results from retrieve_context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Group chunks by section
        sections = {}
        for chunk in results['chunks']:
            section_title = chunk['section_title']
            if section_title not in sections:
                sections[section_title] = []
            sections[section_title].append(chunk)
        
        # Format each section
        for section_title, chunks in sections.items():
            context_parts.append(f"## {section_title}\n")
            
            for chunk in chunks:
                context_parts.append(f"[Page {chunk['page']}] {chunk['text']}\n")
            
            context_parts.append("")  # Empty line between sections
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer a question using RAG
        
        Args:
            question: User question
            
        Returns:
            Dictionary with context and answer placeholder
        """
        # Retrieve relevant context
        print(f"\nQuestion: {question}")
        print("Retrieving context...")
        
        results = self.retrieve_context(question)
        context = self.format_context(results)
        
        print(f"✓ Retrieved {len(results['chunks'])} relevant chunks from {len(results['sections'])} sections")
        
        # In a real RAG system, you would:
        # 1. Format prompt with context + question
        # 2. Send to LLM (GPT-4, Claude, Llama, etc.)
        # 3. Return generated answer
        
        # For this example, we just return the context
        return {
            "question": question,
            "context": context,
            "sections": results['sections'],
            "chunks": results['chunks'],
            "answer": "[LLM would generate answer here based on context]"
        }


def demo_rag_pipeline():
    """Demonstrate the RAG pipeline"""
    print("="*60)
    print("RAG PIPELINE DEMO")
    print("="*60)
    
    try:
        # Initialize pipeline
        rag = RAGPipeline(
            collection_name="test_collection",
            persist_directory="./test_chroma_db"
        )
        
        # Example questions
        questions = [
            "What was the company's revenue performance?",
            "What are the main risk factors?",
            "What is the future outlook and strategy?"
        ]
        
        for question in questions:
            print("\n" + "="*60)
            result = rag.answer_question(question)
            
            print("\nRetrieved Context:")
            print("-"*60)
            print(result['context'][:500] + "..." if len(result['context']) > 500 else result['context'])
            
            print("\nSources:")
            for section in result['sections']:
                print(f"  - {section['title']} (pages {section['pages']})")
            
            print("\n[In production, LLM would generate answer using this context]")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED")
        print("="*60)
        
        print("\nTo integrate with an LLM:")
        print("  1. Install: pip install openai  # or anthropic, etc.")
        print("  2. Format prompt: context + question")
        print("  3. Call LLM API with prompt")
        print("  4. Return generated answer")
        
        print("\nExample prompt format:")
        print("""
        Context:
        {context}
        
        Question: {question}
        
        Answer based only on the context above:
        """)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to run test_vector_store.py first to create the test database")


def example_with_llm():
    """
    Example of how to integrate with an LLM
    (Requires OpenAI API key or similar)
    """
    print("\n" + "="*60)
    print("LLM INTEGRATION EXAMPLE")
    print("="*60)
    
    print("""
# Example with OpenAI GPT-4

from openai import OpenAI

def answer_with_llm(question: str, rag_pipeline: RAGPipeline):
    # Retrieve context
    results = rag_pipeline.retrieve_context(question)
    context = rag_pipeline.format_context(results)
    
    # Format prompt
    prompt = f'''
    Context from financial documents:
    {context}
    
    Question: {question}
    
    Please answer the question based only on the context provided above.
    If the context doesn't contain enough information, say so.
    '''
    
    # Call LLM
    client = OpenAI(api_key="your-api-key")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial analyst assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": results['sections']
    }

# Usage
rag = RAGPipeline()
result = answer_with_llm("What was the revenue growth?", rag)
print(result['answer'])
    """)


def main():
    demo_rag_pipeline()
    example_with_llm()


if __name__ == "__main__":
    main()
