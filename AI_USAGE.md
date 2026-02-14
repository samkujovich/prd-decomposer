# AI Usage Documentation

This document tracks AI tool usage during development of prd-decomposer.

## Tools Used

- **Claude Code** (Anthropic): Project planning, code generation, code review, security hardening
- **GPT-4o** (OpenAI): Runtime LLM for PRD analysis and ticket decomposition

## AI-Generated vs Human-Written

### Fully AI-Generated (with human review)
- `src/prd_decomposer/models.py` - Pydantic model definitions
- `src/prd_decomposer/prompts.py` - LLM prompt templates
- `tests/test_tools.py` - Model unit tests
- `tests/test_server.py` - Server/tool tests with mocked LLM
- `tests/test_prompts.py` - Prompt template tests
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

## Prompts Used

Key prompts used during development:

1. **Project scaffolding**: "Create a PRD decomposer MCP server with arcade-mcp..."
2. **Model generation**: "Create Pydantic models for requirements and Jira tickets..."
3. **Test generation**: "Write pytest tests validating the Pydantic models..."
4. **Code review**: "Review this repo in its entirety and make suggestions as to where to improve..."
5. **Security hardening**: "Address all the issues identified in the code review..."

## Quality Assurance

- All AI-generated code was reviewed before committing
- 51 unit tests with 99% code coverage
- Arcade evals validate tool selection (8 eval cases)
- Security review for path traversal and injection risks
- Error handling review for LLM failure scenarios
- Manual testing with 10 sample PRDs
