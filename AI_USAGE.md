# AI Usage Documentation

This document tracks AI tool usage during development of prd-decomposer.

## Tools Used

- **Claude Code** (Anthropic): Project planning, code generation, code review, security hardening
- **GPT-4o** (OpenAI): Runtime LLM for PRD analysis and ticket decomposition
- **[Excalidraw Plugin for Claude Code](https://github.com/pwood16/claude-kit/tree/main/plugins/excalidraw)**: Architecture diagram generation
- **[Agentic Code Reviewer](https://github.com/richhaase/agentic-code-reviewer/blob/main/CLAUDE.md)**: Code review methodology and principal engineer feedback patterns

## AI-Generated vs Human-Written

### Fully AI-Generated (with human review)
- `src/prd_decomposer/models.py` - Pydantic model definitions (includes AgentContext)
- `src/prd_decomposer/prompts.py` - LLM prompt templates
- `src/prd_decomposer/formatters.py` - Prompt rendering for AI agents
- `src/prd_decomposer/config.py` - Settings class with environment variable support
- `agent/session_state.py` - Agent session state management
- `tests/test_models.py` - Pydantic model unit tests
- `tests/test_server.py` - Server/tool tests with mocked LLM
- `tests/test_circuit_breaker.py` - Circuit breaker tests
- `tests/test_export.py` - Export format tests
- `tests/test_prompts.py` - Prompt template tests
- `tests/test_config.py` - Configuration validation tests
- `tests/test_agent.py` - Agent CLI tests
- `evals/eval_prd_tools.py` - Arcade eval suite (8 eval cases)
- `tests/integration/test_real_api.py` - Real API integration tests
- `docs/diagrams/architecture.*` - Architecture diagram (Excalidraw + SVG)
- `docs/plans/*-agent-executable-tickets-*.md` - Design and implementation plans
- `README.md` - Documentation

### Human-Written with AI Assistance
- `src/prd_decomposer/server.py` - MCP tool implementations (initial structure AI-generated, security/reliability hardening AI-assisted)
- `agent/agent.py` - Agent consumer (based on OpenAI SDK patterns)
- `scripts/run_all_prds.py` - Batch processing script

### Fully Human-Written
- `samples/sample_prd_*.md` - 10 sample PRD documents
- This file (`AI_USAGE.md`)

## Development Sessions

### Session 1: Initial Implementation
- Project scaffolding and structure
- Pydantic model definitions
- Basic MCP tool implementations
- Initial test suite

### Session 2: Security & Reliability Hardening
Claude Code performed a principal engineer code review and implemented:
- **Security**: Path traversal protection, symlink resolution
- **Reliability**: LLM retry with exponential backoff, thread-safe design (removed global state)
- **Observability**: Token usage tracking, prompt versioning
- **Testing**: Expanded from 27 to 51 tests, coverage from 84% to 99%

### Session 3: Configuration, Dependency Injection & Few-Shot Prompts
Claude Code addressed additional code review feedback:
- **Configuration**: Added `Settings` class using pydantic-settings for environment variable support (`PRD_*` prefix)
- **Dependency Injection**: Refactored LLM-calling functions to accept optional `client` and `settings` parameters for improved testability
- **Few-Shot Examples**: Added input/output examples to both prompts demonstrating ambiguity detection and story decomposition
- **Bounds Validation**: Added constraints to retry config (max_retries 1-10, initial_retry_delay 0-60s)
- **MCP Compatibility Fix**: Changed `decompose_to_tickets` parameter from `dict` to JSON string for better agent compatibility
- **Testing**: Expanded from 51 to 67 tests

### Session 4: Export Formats, Circuit Breaker & Ambiguity Enhancements
Claude Code implemented major feature additions from code review:
- **Export Formats**: Added `export_tickets` tool supporting CSV, Jira REST API, and YAML output
- **Circuit Breaker Pattern**: Implemented thread-safe circuit breaker for upstream failure protection (closed→open→half_open states)
- **Configurable Sizing Rubric**: Added `SizingRubric` model with customizable S/M/L definitions
- **Actionable Ambiguity Flags**: Enhanced `AmbiguityFlag` model with `severity` and `suggested_action` fields
- **Health Check Endpoint**: Added `health_check` tool for service status monitoring
- **Structured Logging**: Added JSON logging with correlation IDs (`log.py`)
- **Testing**: Expanded from 67 to 132 tests

### Session 5: Bug Fixes Round 1-3
Claude Code fixed bugs identified in code review:
- **Circuit Breaker Fixes**: Half-open slot leak prevention, proper failure recording for non-retryable errors
- **YAML Export Fixes**: Single `epics:` key, proper escaping for quotes/newlines/backslashes, per-item list quoting
- **Sizing Rubric Fixes**: Auto-fill labels validator, non-object JSON validation
- **Export Validation**: Added type checks for non-object JSON inputs
- **Testing**: Expanded from 132 to 206 tests

### Session 6: Bug Fixes Round 4-6
Claude Code fixed additional edge cases:
- **Circuit Breaker Refinements**:
  - 4xx client errors don't trip breaker (only transient upstream failures)
  - Half-open probes limited to single attempt (no retries)
  - Half-open detection after open→half_open transition
  - Retry backoff uses effective attempt limit
- **Health Endpoint**: Consistent "degraded" status for both open and half_open states
- **Export Validation**: Deep nested validation for stories array and story objects
- **Jira Export**: Moved metadata outside `issueUpdates` to comply with Jira schema
- **YAML Export**: Consistent empty collection handling (`[]` instead of null/omitted)
- **Testing**: Expanded from 206 to 230 tests (unit)

### Session 7: Logging Initialization & Integration Tests
Claude Code completed final polish items:
- **Logging Initialization**: Call `setup_logging(get_settings())` at server startup for structured JSON logs
- **Integration Tests**: Added `tests/integration/` with 4 real API tests (skipped when `OPENAI_API_KEY` not set)
- **Documentation Review**: Updated README and AI_USAGE for accuracy
- **Testing**: Expanded from 230 to 234 tests (230 unit + 4 integration)

### Session 8: Bug Fixes Round 7-9 (Code Review)
Claude Code fixed 11 bugs identified in code review:
- **Jira Priority Mapping**: Handle non-string priority values (int, bool, None) by coercing to string
- **Nested Field Validation**: Add `_validate_string_list` helper to coerce/filter story field arrays
- **Jira Schema Compliance**: Remove `_prd_decomposer_metadata` from Jira bulk payload (breaks REST API)
- **Circuit Breaker Half-Open 4xx**: 4xx during half-open probe closes circuit (upstream is responsive)
- **Circuit Breaker Closed 4xx**: 4xx in closed state resets failure streak (5xx/4xx/5xx not 3 consecutive failures)
- **Scalar Field Validation**: Add `_validate_string_field` helper for title/description validation
- **Shutdown Circuit Breaker**: Move shutdown check before `llm_call_attempted` flag to prevent false failures
- **YAML Size/Priority Escaping**: Quote and escape size/priority fields in YAML export
- **Empty String Validation**: Reject empty strings for required fields (title)
- **Lazy Logging Init**: Move `setup_logging()` from import-time to runtime via `_ensure_logging_initialized()`
- **Null Stories Handling**: Normalize `stories: null` to `[]` during export validation
- **Testing**: Expanded from 234 to 263 tests (259 unit + 4 integration)

### Session 9: Production Hardening & Test Reorganization
Claude Code addressed final code review feedback:
- **Production Hardening**: Moved `_logging_initialized` reset to conftest, removed misleading `__all__`, added exception chaining with `from e`, switched to `time.monotonic()` for elapsed time, reverted to single user message pattern for LLM calls
- **Export Cleanup**: Changed `_comment` to `_metadata` in YAML export, simplified `_map_priority_to_jira` to trust Pydantic validation
- **Test Reorganization**: Extracted circuit breaker tests to `test_circuit_breaker.py` and export tests to `test_export.py` to mirror source structure per CLAUDE.md conventions
- **Testing**: 243 tests (239 unit + 4 integration) after reorganization

### Session 10: Agent-Executable Tickets Feature
Claude Code implemented AI-optimized ticket output:
- **AgentContext Model**: Added `AgentContext` Pydantic model with `goal`, `exploration_paths`, `exploration_hints`, `known_patterns`, `verification_tests`, and `self_check` fields
- **Story Model Update**: Added optional `agent_context` field to `Story` model (backward compatible)
- **Prompt Update**: Updated `DECOMPOSE_TO_TICKETS_PROMPT` with guideline 8 for agent_context generation, including few-shot example
- **Prompt Renderer**: Created `src/prd_decomposer/formatters.py` with `render_agent_prompt()` function for copy-paste prompts
- **CLI Command**: Added `prompt N` command (with `copy N` and `show N` aliases) to agent CLI
- **CSV Export**: Added `agent_prompt` column to CSV export format
- **Architecture Fix**: Moved `render_agent_prompt` to server package to maintain proper dependency direction (agent imports from server, not vice versa)
- **Testing**: Expanded from 243 to 306 tests

### Session 11: Export Enhancements & Agent Improvements
Claude Code enhanced exports and agent behavior:
- **Jira Export**: Added "AI Implementation Context" section to story descriptions with goal, exploration paths, hints, patterns, verification tests, and self-check questions
- **YAML Export**: Added full `agent_context` object to story output
- **Agent Import Fix**: Added try/except to handle both module and script import styles (`from agent.formatters` vs `from formatters`)
- **Agent Instructions**: Updated to show full agent_context when presenting tickets (all 6 fields: Goal, Exploration paths, Start with, Patterns, Verification, Self-check)
- **Quality Evals**: Added `TestAgentContextGeneration` class with 5 evals for agent_context output quality
- **Testing**: Expanded from 306 to 310 tests

## Prompts Used

Key prompts used during development:

1. **Project scaffolding**: "Create a PRD decomposer MCP server with arcade-mcp..."
2. **Model generation**: "Create Pydantic models for requirements and Jira tickets..."
3. **Test generation**: "Write pytest tests validating the Pydantic models..."
4. **Code review**: See full prompt below
5. **Security hardening**: "Address all the issues identified in the code review..."

### Code Review Prompt (Principal Engineer / CTO / CPO)

```
Review this codebase as if you were a Principal Engineer, CTO, and CPO all reviewing the same PR. Give critical feedback on:

- Architecture and design decisions
- Testability and test coverage gaps
- Security vulnerabilities or risks
- Production-readiness concerns
- Code quality and maintainability
- Missing error handling or edge cases

Don't hold back. Be direct about what needs to improve before this ships.
```

This prompt pattern is adapted from the [Agentic Code Reviewer](https://github.com/richhaase/agentic-code-reviewer/blob/main/CLAUDE.md) methodology.

## Quality Assurance

- All AI-generated code was reviewed before committing
- 310 tests (306 unit + 4 integration) with comprehensive coverage
- Arcade evals validate tool selection (8 eval cases)
- Output quality evals validate agent_context generation (21 eval cases)
- Security review for path traversal and injection risks
- Error handling review for LLM failure scenarios
- Manual testing with 10 sample PRDs
- End-to-end testing with OpenAI Agents SDK agent
