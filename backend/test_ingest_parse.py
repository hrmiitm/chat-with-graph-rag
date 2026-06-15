import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.llm_client import generate, parse_json_from_llm
from app.prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_PROMPT

chunk_text = """ACME Corporation — Employee Leave Policy
Effective Date: January 1, 2025
Version: 2.1

1. Annual Leave
All full-time employees are entitled to 21 days of paid annual leave per calendar year. Part-time employees receive leave proportional to their working hours. Annual leave must be approved by the direct manager at least 5 business days in advance. Unused leave can be carried over to the next year, up to a maximum of 10 days.

2. Sick Leave
Employees are entitled to 12 days of paid sick leave per year. A me"""

prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk_text)
print("Sending request to vLLM...")
res = generate(prompt, system=ENTITY_EXTRACTION_SYSTEM)
print("RAW LENGTH:", len(res['text']))
print("RAW TEXT:")
print(res['text'])
print("=" * 40)
try:
    data = parse_json_from_llm(res['text'])
    print("SUCCESSFULLY PARSED:", type(data))
    print(data)
except Exception as e:
    print("PARSING ERROR:", type(e), str(e))
