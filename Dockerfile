# Use a Python image with build tools already included
FROM python:3.11

WORKDIR /app

# No apt-get needed - python:3.11 already has build tools

RUN pip install --upgrade pip && \
    pip install "numpy>=1.26.0,<2.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/MinishLab/model2vec.git && \
    pip install --force-reinstall "chonkie[model2vec]"

# Copy all application files
COPY json_text_processor.py .
COPY chunking_comparison.py .
COPY main_pipeline.py .
COPY neo4j_knowledge_graph.py .
COPY build_knowledge_graph.py .
COPY update_knowledge_graph.py .
COPY graphrag_system.py .
COPY test_embeddings.py .
COPY test_simple_embeddings.py .
COPY test_chunking_models.py .
COPY test_embedding_models.py .

# Create directories for input, output, and samples
RUN mkdir -p /app/input /app/output /app/samples

ENV PYTHONUNBUFFERED=1

# Use the new pipeline script that handles JSON processing and chunking
CMD ["python", "main_pipeline.py"]