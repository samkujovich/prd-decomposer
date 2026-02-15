"""Prompt templates for PRD analysis and decomposition."""

# Version for traceability - increment when prompts change
PROMPT_VERSION = "1.6.0"

# Default sizing rubric text (used when no custom rubric provided)
DEFAULT_SIZING_RUBRIC = """   - S (Small): Less than 1 day, Single component, Low risk
   - M (Medium): 1-3 days, May touch multiple components, Moderate complexity
   - L (Large): 3-5 days, Significant complexity, Unknowns or cross-team coordination"""

ANALYZE_PRD_PROMPT = """You are a senior technical product manager. Analyze the following PRD and extract structured requirements.

For each requirement you identify:
1. Assign a unique ID (REQ-001, REQ-002, etc.)
2. Write a clear title and description
3. Extract or infer acceptance criteria (testable conditions for success)
4. Identify dependencies on other requirements (by ID)
5. Flag ambiguities with STRUCTURED, ACTIONABLE feedback:
   - Categories: missing_acceptance_criteria, vague_quantifier, undefined_term, missing_details, conflicting_requirements, out_of_scope, security_concern, other
   - Severity: critical (blocks implementation), warning (should address before sprint), suggestion (nice to clarify)
   - Always provide a specific suggested_action the PRD author can take
6. Assign priority: "high", "medium", or "low" based on language cues and business impact

## Example

**Input PRD:**
# Feature: Password Reset
Users must be able to reset their password via email.
The reset flow should be fast and user-friendly.

**Output:**
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "Email-based password reset",
      "description": "Users can request a password reset link sent to their registered email address",
      "acceptance_criteria": [
        "User can request reset from login page",
        "Reset email sent within 30 seconds",
        "Reset link expires after 1 hour",
        "User can set new password via reset link"
      ],
      "dependencies": [],
      "ambiguity_flags": [
        {{
          "category": "vague_quantifier",
          "issue": "'fast' has no specific latency requirement defined",
          "severity": "warning",
          "suggested_action": "Define target latency, e.g., 'Reset email sent within 30 seconds of request'"
        }},
        {{
          "category": "vague_quantifier",
          "issue": "'user-friendly' has no measurable UX criteria specified",
          "severity": "suggestion",
          "suggested_action": "Add testable UX criteria, e.g., 'Password reset completes in under 3 clicks'"
        }}
      ],
      "priority": "high"
    }}
  ],
  "summary": "Password reset feature allowing users to recover account access via email"
}}

---

Now analyze this PRD:

<prd_document>
{prd_text}
</prd_document>

Return valid JSON matching this exact schema:
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "string",
      "description": "string",
      "acceptance_criteria": ["string"],
      "dependencies": ["REQ-XXX"],
      "ambiguity_flags": [
        {{
          "category": "missing_acceptance_criteria|vague_quantifier|undefined_term|missing_details|conflicting_requirements|out_of_scope|security_concern|other",
          "issue": "Description of the ambiguity",
          "severity": "critical|warning|suggestion",
          "suggested_action": "Specific action the PRD author should take to resolve this"
        }}
      ],
      "priority": "high|medium|low"
    }}
  ],
  "summary": "Brief 1-2 sentence overview of the PRD",
  "source_hash": "Will be set by the system"
}}"""


DECOMPOSE_TO_TICKETS_PROMPT = """You are a senior engineering manager. Convert these structured requirements into Jira-ready epics and stories.

Guidelines:
1. Group related requirements into epics (1-4 epics typically)
2. Break each requirement into implementable stories (1-3 stories per requirement)
3. Size stories using this rubric:
{sizing_rubric}
4. Set story priority based on source requirement priority ("high", "medium", "low")
5. Generate descriptive labels (e.g., "backend", "frontend", "api", "database", "auth", "testing")
6. Preserve traceability by including requirement_ids on each story
7. Write clear acceptance criteria derived from the requirements
8. For each story, include an agent_context object with:
   - goal: A clear statement of WHY this work matters (the problem being solved)
   - exploration_paths: Keywords/concepts an AI agent should search for to understand context
   - exploration_hints: Specific file paths or modules to start with (if inferrable from requirements)
   - known_patterns: Libraries, conventions, or existing code patterns to follow
   - verification_tests: Test names or commands to verify completion
   - self_check: Questions to validate edge cases, security, and correctness

## Example

**Input Requirements:**
{{
  "requirements": [
    {{
      "id": "REQ-001",
      "title": "Email-based password reset",
      "description": "Users can request a password reset link sent to their email",
      "acceptance_criteria": ["Reset email sent within 30 seconds", "Link expires after 1 hour"],
      "dependencies": [],
      "ambiguity_flags": [],
      "priority": "high"
    }}
  ],
  "summary": "Password reset feature"
}}

**Output:**
{{
  "epics": [
    {{
      "title": "Password Reset",
      "description": "Enable users to securely reset their passwords via email",
      "stories": [
        {{
          "title": "Create password reset request endpoint",
          "description": "Implement POST /auth/reset-password endpoint that validates email and sends reset link",
          "acceptance_criteria": [
            "Endpoint accepts email in request body",
            "Returns 200 for valid registered emails",
            "Returns 200 for unregistered emails (prevent enumeration)",
            "Triggers email send within 30 seconds"
          ],
          "size": "M",
          "priority": "high",
          "labels": ["backend", "api", "auth"],
          "requirement_ids": ["REQ-001"],
          "agent_context": {{
            "goal": "Allow users who forgot their password to securely regain account access without contacting support",
            "exploration_paths": ["password reset", "authentication", "email sending", "token generation"],
            "exploration_hints": ["src/auth/", "src/email/"],
            "known_patterns": ["Use existing email service", "Follow JWT token pattern for reset tokens"],
            "verification_tests": ["test_password_reset_request", "test_reset_token_expiry"],
            "self_check": [
              "Does this prevent email enumeration attacks?",
              "Is the reset token cryptographically secure?",
              "What happens if the email service is down?"
            ]
          }}
        }},
        {{
          "title": "Implement password reset email template",
          "description": "Create email template with secure reset link and branding",
          "acceptance_criteria": [
            "Email contains secure one-time reset link",
            "Link expires after 1 hour",
            "Email follows brand guidelines"
          ],
          "size": "S",
          "priority": "high",
          "labels": ["backend", "email"],
          "requirement_ids": ["REQ-001"]
        }},
        {{
          "title": "Build password reset form UI",
          "description": "Create frontend form for entering new password after clicking reset link",
          "acceptance_criteria": [
            "Form validates password strength",
            "Shows success/error states",
            "Redirects to login on success"
          ],
          "size": "M",
          "priority": "high",
          "labels": ["frontend", "auth"],
          "requirement_ids": ["REQ-001"]
        }}
      ],
      "labels": ["auth", "security"]
    }}
  ],
  "metadata": {{
    "requirement_count": 1,
    "story_count": 3
  }}
}}

---

<requirements_document>
{requirements_json}
</requirements_document>

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
          "priority": "high|medium|low",
          "labels": ["string"],
          "requirement_ids": ["REQ-XXX"],
          "agent_context": {{
            "goal": "string",
            "exploration_paths": ["string"],
            "exploration_hints": ["string"],
            "known_patterns": ["string"],
            "verification_tests": ["string"],
            "self_check": ["string"]
          }}
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
}}"""
