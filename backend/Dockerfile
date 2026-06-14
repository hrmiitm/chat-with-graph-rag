# FastAPI Backend for GraphRAG
# Runs uvicorn on port 8001

FROM docker.io/library/python:3.11-slim

WORKDIR /app

# Install system deps for PyMuPDF and psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
