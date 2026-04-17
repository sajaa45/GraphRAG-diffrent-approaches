#!/usr/bin/env python3
"""Check what sections are in ChromaDB"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="financial_docs")

# Get all sections
results = collection.get(
    where={"type": "section"},
    include=["metadatas", "documents"]
)

print(f"Total sections in ChromaDB: {len(results['ids'])}\n")
print("="*80)

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"\nSection {i}:")
    print(f"  Title: {meta.get('title', 'N/A')}")
    print(f"  Section ID: {meta.get('section_id', 'N/A')}")
    print(f"  Preview: {doc[:200]}...")
    print("-"*80)

# Test OPERATES_IN keywords
print("\n" + "="*80)
print("TESTING OPERATES_IN KEYWORDS")
print("="*80)

model = SentenceTransformer("all-MiniLM-L6-v2")
test_keywords = "business overview operations segments market industry"
embedding = model.encode([test_keywords])[0]

section_results = collection.query(
    query_embeddings=[embedding.tolist()],
    n_results=5,
    where={"type": "section"}
)

print(f"\nTop 5 sections for '{test_keywords}':\n")
for i, (doc, meta, dist) in enumerate(zip(
    section_results['documents'][0],
    section_results['metadatas'][0],
    section_results['distances'][0]
), 1):
    similarity = 1 - dist
    print(f"{i}. {meta.get('title', 'N/A')} (similarity: {similarity:.3f})")
    print(f"   Section ID: {meta.get('section_id', 'N/A')}")
    print(f"   Preview: {doc[:150]}...")
    print()
