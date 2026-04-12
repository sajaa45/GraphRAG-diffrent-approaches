#!/usr/bin/env python3
"""
Simple web interface to browse and query the vector store
"""

from flask import Flask, render_template_string, request, jsonify
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import json

app = Flask(__name__)

# Initialize vector store
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="financial_docs")
embedding_model = SentenceTransformer("s