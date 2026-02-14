"""Prompt templates for PRD analysis and decomposition."""

ANALYZE_PRD_PROMPT = '''You are a senior technical product manager. Analyze the following PRD and extract structured requirements.

For each requirement you identify:
1. Assign a unique ID (REQ-001, REQ-002, etc.)
2. Write a clear title and description
3. Extract or infer acceptance criteria (testable conditions for success)
4. Identify dependencies on other requirements (by ID)
5. Flag ambiguities - add to ambiguity_flags if:
   - Missing acceptance criteria (no clear way to test success)
   - Vague quantifiers without metrics (e.g., "fast", "scalable", "user-friendly", "easy to use")
6. Assign priority: "high", "medium", or "low" based on language cues and business impact

PRD:
{prd_text}

Return valid JSON matching this exact schema:
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "string",
      "description": "string",
      "acceptance_criteria": ["string"],
      "dependencies": ["REQ-XXX"],
      "ambiguity_flags": ["string describing the ambiguity"],
      "priority": "high|medium|low"
    }}
  ],
  "summary": "Brief 1-2 sentence overview of the PRD",
  "source_hash": "Use first 8 chars of a hash of the PRD text"
}}'''


DECOMPOSE_TO_TICKETS_PROMPT = '''You are a senior engineering manager. Convert these structured requirements into Jira-ready epics and stories.

Guidelines:
1. Group related requirements into epics (1-4 epics typically)
2. Break each requirement into implementable stories (1-3 stories per requirement)
3. Size stories using this rubric:
   - S (Small): Less than 1 day, single component, low risk
   - M (Medium): 1-3 days, may touch multiple components, moderate complexity
   - L (Large): 3-5 days, significant complexity, unknowns, or cross-team coordination
4. Generate descriptive labels (e.g., "backend", "frontend", "api", "database", "auth", "testing")
5. Preserve traceability by including requirement_ids on each story
6. Write clear acceptance criteria derived from the requirements

Requirements:
{requirements_json}

Return valid JSON matching this exact schema:
{{
  "epics": [
    {{
      "title": "string",
      "description": "string",
      "stories": [
        {{
          "title": "string",
          "description": "string",
          "acceptance_criteria": ["string"],
          "size": "S|M|L",
          "labels": ["string"],
          "requirement_ids": ["REQ-XXX"]
        }}
      ],
      "labels": ["string"]
    }}
  ],
  "metadata": {{
    "generated_at": "ISO timestamp",
    "model": "gpt-4o",
    "requirement_count": number,
    "story_count": number
  }}
}}'''
