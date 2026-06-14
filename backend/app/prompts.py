"""
All LLM prompt templates in one place.
Simple f-string templates — easy to read, easy to edit.
"""

# =============================================================================
# Entity & Relation Extraction
# =============================================================================
ENTITY_EXTRACTION_SYSTEM = """You are a precise information extraction assistant.
Extract entities and relations from the given text.
Return valid JSON only, no markdown, no explanation."""

ENTITY_EXTRACTION_PROMPT = """Extract all important entities and relationships from this text.

TEXT:
{chunk_text}

Return JSON in this exact format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Person|Organization|Concept|Location|Event|Policy|Technology|Other"}}
  ],
  "relations": [
    {{"source": "Entity A", "relation": "relates_to|part_of|created_by|located_in|causes|requires|manages|other", "target": "Entity B"}}
  ]
}}

Rules:
- Extract real, meaningful entities (not generic words like "system" or "process")
- Relations should be factual and stated in the text
- Keep entity names concise (1-4 words)
- Return empty lists if no clear entities/relations exist"""


# =============================================================================
# Answer Generation (GraphRAG)
# =============================================================================
ANSWER_SYSTEM = """You are a helpful assistant that answers questions based on provided context.
Always cite which source chunks support your answer.
If the context doesn't contain enough information, say so honestly."""

ANSWER_PROMPT = """Answer the user's question using ONLY the context provided below.

CONTEXT CHUNKS (from document search):
{context_chunks}

KNOWLEDGE GRAPH FACTS:
{graph_facts}

USER QUESTION: {question}

Instructions:
- Use the context chunks and graph facts to answer accurately
- Cite sources using [Chunk N] notation
- If multiple chunks support a point, cite all of them
- If the context doesn't contain the answer, say "I don't have enough information to answer this based on the uploaded documents."
- Be concise but thorough"""


# =============================================================================
# Question Entity Extraction (for graph search)
# =============================================================================
QUESTION_ENTITY_PROMPT = """Extract key entities from this question that should be searched in a knowledge graph.
Return only a JSON list of entity names, nothing else.

Question: {question}

Example output: ["entity1", "entity2"]"""
