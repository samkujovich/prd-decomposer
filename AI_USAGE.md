# AI Usage Documentation

This document tracks AI tool usage during development of prd-decomposer.

## Tools Used

- **Claude Code** (Anthropic): Project planning, code generation, documentation
- **GPT-4o** (OpenAI): Runtime LLM for PRD analysis and ticket decomposition

## AI-Generated vs Human-Written

### Fully AI-Generated (with human review)
- `src/prd_decomposer/models.py` - Pydantic model definitions
- `src/prd_decomposer/prompts.py` - LLM prompt templates
- `tests/test_tools.py` - Unit test scaffolding
- `evals/eval_prd_tools.py` - Arcade eval suite
- `README.md` - Documentation

### Human-Written with AI Assistance
- `src/prd_decomposer/server.py` - MCP tool implementations (structure AI-generated, logic reviewed)
- `agent/agent.py` - Agent consumer (based on OpenAI SDK patterns)

### Fully Human-Written
- `samples/sample_prd.md` - Sample PRD content
- This file (`AI_USAGE.md`)

## Prompts Used

Key prompts used during development:

1. **Project scaffolding**: "Create a PRD decomposer MCP server with arcade-mcp..."
2. **Model generation**: "Create Pydantic models for requirements and Jira tickets..."
3. **Test generation**: "Write pytest tests validating the Pydantic models..."

## Quality Assurance

- All AI-generated code was reviewed before committing
- Unit tests validate model behavior
- Arcade evals validate tool selection
- Manual testing with sample PRD
