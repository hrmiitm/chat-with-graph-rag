# Final framework

```text
FastAPI
+ PostgreSQL 15
+ pgvector
+ Apache AGE
+ vLLM
+ Qwen3
+ Qwen3 Embedding / Reranker
+ OpenAI-compatible SDK
+ Streamlit demo UI
+ Docker Compose
```

## Why this stack is final

PostgreSQL is safe for commercial projects because it uses a liberal open-source license, and its official FAQ says there is no fee even for commercial software products. Apache AGE adds graph database functionality inside PostgreSQL and is Apache 2.0 licensed. pgvector stores embeddings and supports vector similarity search inside PostgreSQL. vLLM gives you a local OpenAI-compatible LLM server, so your backend can call local models without cloud APIs. Qwen3 has dense models like 4B, 8B, 14B, 32B under Apache 2.0, and Qwen3 Embedding/Reranker models are also Apache 2.0. ([PostgreSQL][1])

---

# Final tech choice

| Part                    | Final choice                                              |
| ----------------------- | --------------------------------------------------------- |
| Backend                 | **FastAPI**                                               |
| AI call layer           | **OpenAI-compatible SDK/client**                          |
| LLM serving             | **vLLM**                                                  |
| Local laptop model      | **Qwen3-4B**                                              |
| AWS / big machine model | **Qwen3-14B or Qwen3-32B**                                |
| Embedding               | **Qwen3-Embedding-0.6B** first                            |
| Reranker                | **Qwen3-Reranker-0.6B** first                             |
| Database                | **PostgreSQL 15**                                         |
| Vector search           | **pgvector**                                              |
| Knowledge graph         | **Apache AGE**                                            |
| UI demo                 | **Streamlit**                                             |
| Storage                 | local folder first, **MinIO** later                       |
| Background jobs         | simple FastAPI task first, **Celery + Redis** later       |
| Deployment              | **Docker Compose** first, Kubernetes later only if needed |

FastAPI is MIT licensed and described as ready for production; Streamlit is Apache 2.0 licensed, so both are fine for this commercial-demo path. ([GitHub][2])

---

# How everything works together

```text
User uploads PDF/DOCX/TXT
        ↓
FastAPI receives file
        ↓
Text is extracted and split into chunks
        ↓
Qwen3-Embedding creates embeddings
        ↓
Chunks + embeddings stored in PostgreSQL using pgvector
        ↓
Qwen3 extracts entities and relations
        ↓
Entities + relations stored in Apache AGE graph
        ↓
User asks a question
        ↓
Question embedding searches pgvector
        ↓
Question entities search Apache AGE graph
        ↓
Vector context + graph facts are merged
        ↓
Qwen3 generates final answer with citations
```

This is clean GraphRAG:

```text
pgvector = semantic memory
Apache AGE = relationship memory
PostgreSQL = production data system
vLLM = local on-prem brain
FastAPI = product API
Streamlit = Frontend UI / demo application
```

---

# Stage 1: local laptop demo

Use this first:

```text
FastAPI
PostgreSQL + pgvector + Apache AGE
vLLM with Qwen3-4B
Qwen3-Embedding-0.6B
Streamlit UI
Docker Compose
```

- Make sure to do with functional and excellent clean code like senior software engineer
- Make proper documentation for each code.
- Do not create lot of files etc, i want clean easily extensible code and easily undestandable.
- Use podman or podman compose so that laptop demo is clean and easily up down, and fully working  without installation issues.


What to show locally:

```text
1. Upload document
2. Ingest document
3. Show chunks created
4. Show entities extracted
5. Show graph relations
6. Ask question
7. Show answer + source chunks + graph facts
```

For local laptop, keep it simple:

```text
LLM: Qwen3-4B
Embedding: Qwen3-Embedding-0.6B
Reranker: optional
Documents: 1-3 sample PDFs of 1-2 pages
Users: two user demo
```

- make sure in stream lit ui, all the pipeline is very easily debugable, and undesratangble, 
- show all the chunks, their embeddings, retrieved chunks, scores, reranking, total cost, overall graph visulizaton ....
- think of reranking bmr25 and cosine similarity, and show the scores, and show the final answer with citations, and also use vector and hybrid search. I mean to say not just graph rag, however it can be dragastically imporve by using hybrid or hyde or bm25  or reranking, then use them as well
- i feel rereanking will defineltl improve performance so use that as well, and show in frontend as well

---

# Stage 2: AWS production-looking demo

Use the **same code**, only change deployment size:

```text
FastAPI backend container
PostgreSQL container or managed PostgreSQL-compatible VM
vLLM on GPU instance
Streamlit frontend
Nginx reverse proxy
Docker Compose
```

For AWS demo:

```text
LLM: Qwen3-14B
Embedding: Qwen3-Embedding-0.6B or 4B
Reranker: Qwen3-Reranker-0.6B
Users: multiple browser sessions
Auth: simple username/password or Keycloak later
```

Key features/demonstration points:

```text
1. Multiple users can chat
2. Each user has separate chat history
3. Documents are indexed once
4. Answers use local AWS GPU model
5. No OpenAI/Claude/Gemini API call
6. Same stack can move to on-prem
```

---

# Stage 3: Production / on-premise deployment

On a dedicated local GPU server:

```text
GPU server
Docker Compose or Kubernetes
PostgreSQL data volume
vLLM model server
FastAPI backend
Streamlit/React frontend
Local file storage or MinIO
```

For real on-prem:

```text
LLM: Qwen3-14B / Qwen3-32B
Embedding: Qwen3-Embedding-4B if GPU allows
Reranker: Qwen3-Reranker-0.6B or 4B
Database: PostgreSQL with backup
Auth: Keycloak or internal login
Monitoring: Prometheus + Grafana later
```

---

# Final repo structure

```text
graph-rag-production-demo/
  docker-compose.yml
  .env.example
  README.md
  THIRD_PARTY_LICENSES.md

  backend/
    app/
      main.py
      config.py

      api/
        documents.py
        chat.py
        graph.py
        health.py

      ingestion/
        loader.py
        chunker.py
        embedder.py
        graph_extractor.py
        ingest_service.py

      retrieval/
        vector_search.py
        graph_search.py
        reranker.py
        context_builder.py
        answer_generator.py

      llm/
        client.py
        prompts.py

      db/
        postgres.py
        schema.sql
        age_queries.py

  ui/
    streamlit_app.py

  sample_docs/
    demo_policy.pdf
    demo_manual.pdf
```

---

# Final build order

Build in this order only:

```text
1. Start PostgreSQL with pgvector + AGE
2. Start vLLM with Qwen3-4B
3. Build FastAPI health endpoint
4. Build document upload
5. Build text extraction + chunking
6. Store chunks + embeddings in pgvector
7. Extract triples using Qwen3
8. Store graph in Apache AGE
9. Build chat endpoint
10. Build Streamlit UI
11. Add Docker Compose
12. Add AWS deployment
```

---

# My final decision

Use this final stack:

```text
Backend:
FastAPI

AI calls:
OpenAI-compatible SDK/client

LLM serving:
vLLM

Model:
Qwen3-4B locally
Qwen3-14B/32B on AWS and on-prem

Embedding:
Qwen3-Embedding-0.6B first

Reranking:
Qwen3-Reranker-0.6B optional but recommended

Database:
PostgreSQL 15

Vector:
pgvector

Graph:
Apache AGE

UI:
Streamlit for demo
React later only if needed

Deployment:
Docker Compose for laptop, AWS, and on-prem
```

Project summary:

> A fully on-premise GraphRAG system where PostgreSQL stores documents, vectors, graph relations, chat history, and audit data; Apache AGE provides the knowledge graph; pgvector provides semantic search; vLLM serves Qwen3 locally; and FastAPI exposes the production API without any external LLM calls.

