# AI Usage Documentation

This document tracks AI tool usage during development of prd-decomposer.

## Tools Used

- **Claude Code** (Anthropic): Project planning, code generation, code review, security hardening
- **GPT-4o** (OpenAI): Runtime LLM for PRD analysis and ticket decomposition

## AI-Generated vs Human-Written

### Fully AI-Generated (with human review)
- `src/prd_decomposer/models.py` - Pydantic model definitions
- `src/prd_decomposer/prompts.py` - LLM prompt templates
- `src/prd_decomposer/config.py` - Settings class with environment variable support
- `tests/test_tools.py` - Model unit tests
- `tests/test_server.py` - Server/tool tests with mocked LLM
- `tests/test_prompts.py` - Prompt template tests
- `tests/test_config.py` - Configuration validation tests
- `evals/eval_prd_tools.py` - Arcade eval suite (8 eval cases)
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

## Prompts Used

Key prompts used during development:

1. **Project scaffolding**: "Create a PRD decomposer MCP server with arcade-mcp..."
2. **Model generation**: "Create Pydantic models for requirements and Jira tickets..."
3. **Test generation**: "Write pytest tests validating the Pydantic models..."
4. **Code review**: "Review this repo in its entirety and make suggestions as to where to improve..."
5. **Security hardening**: "Address all the issues identified in the code review..."

## Quality Assurance

- All AI-generated code was reviewed before committing
- 67 unit tests with 98% code coverage
- Arcade evals validate tool selection (8 eval cases)
- Security review for path traversal and injection risks
- Error handling review for LLM failure scenarios
- Manual testing with 10 sample PRDs
- End-to-end testing with OpenAI Agents SDK agent
