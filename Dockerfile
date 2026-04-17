# Use a Python image with build tools already included
FROM python:3.11

WORKDIR /app

# No apt-get needed - python:3.11 already has build tools

RUN pip install --upgrade pip && \
    pip install "numpy>=1.26.0,<2.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir google-genai


# Copy all application files
COPY json_text_processor.py .
COPY chunking.py .
COPY main_pipeline.py .
COPY neo4j_knowledge_graph.py .
COPY build_knowledge_graph.py .
COPY update_knowledge_graph.py .
COPY graphrag_system.py .
COPY test_embeddings.py .
COPY test_simple_embeddings.py .
COPY test_chunking_models.py .
COPY test_embedding_models.py .
COPY parse_pdf.py .
COPY content_based_sectioning.py .
COPY chunk_sections.py .
COPY vector_store_pipeline.py .
COPY unified_pipeline.py .
COPY query_vector_store.py .
COPY build_kg_from_query.py .
COPY relation_extraction_config.py .
COPY industry_to_sic.py .
COPY multi_relation_kg_builder.py .

# Create directories for input, output, and samples
RUN mkdir -p /app/input /app/output /app/samples

ENV PYTHONUNBUFFERED=1

# Use the new pipeline script that handles JSON processing and chunking
CMD ["python", "main_pipeline.py"]