"""
All LLM prompt templates in one place.
Simple f-string templates — easy to read, easy to edit.

NOTE: /no_think is a Qwen3 directive that suppresses <think>...</think> reasoning
tags in the output. We strip them in llm_client.py too, but this avoids wasted tokens.
"""

# =============================================================================
# Entity & Relation Extraction
# =============================================================================
ENTITY_EXTRACTION_SYSTEM = """You are a precise information extraction assistant.
Extract entities and relations from the given text.
Return valid JSON only, no markdown, no explanation, no reasoning.
/no_think"""

ENTITY_EXTRACTION_PROMPT = """Extract ALL entities and relationships from this text.

TEXT:
{chunk_text}

Return JSON in this exact format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Person|Organization|Concept|Location|Event|Policy|Technology|Other"}}
  ],
  "relations": [
    {{"source": "Entity A", "relation": "relationship_type", "target": "Entity B"}}
  ]
}}

Rules:
- Extract EVERY named entity: people, organizations, policies, concepts, technologies, locations, amounts, dates
- Create a relation for EVERY factual connection stated or implied between entities
- Use specific relation types: manages, approves, requires, provides, part_of, entitled_to, reports_to, has_limit, covers, specifies, uses, developed_by, monitors, integrates_with
- Keep entity names concise (1-4 words)
- Each chunk should produce at least 3-5 entities and 3-5 relations if the text is informative
- Return empty lists ONLY if the text has no meaningful content

/no_think"""


# =============================================================================
# Answer Generation (GraphRAG)
# =============================================================================
ANSWER_SYSTEM = """You are a helpful assistant that answers questions based on provided context.
Always cite which source chunks support your answer.
If the context doesn't contain enough information, say so honestly.
/no_think"""

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
- Be concise but thorough

/no_think"""


# =============================================================================
# Question Entity Extraction (for graph search)
# =============================================================================
QUESTION_ENTITY_PROMPT = """Extract key entities from this question that should be searched in a knowledge graph.
Return only a JSON list of entity names, nothing else.

Question: {question}

Example output: ["entity1", "entity2"]

/no_think"""
