"""PRD Decomposer evaluation suites.

This package contains two types of evaluations:

1. **Tool Selection Evals** (eval_prd_tools.py)
   - Tests whether the LLM selects the correct tool for a given user intent
   - Uses arcade_evals framework with SimilarityCritic/BinaryCritic
   - Run with: `uv run arcade evals evals/eval_prd_tools.py`

2. **Output Quality Evals** (eval_output_quality.py)
   - Tests whether tool outputs are correct and high-quality
   - Validates ambiguity detection, traceability, sizing, etc.
   - Run with: `uv run pytest evals/eval_output_quality.py -v`

Both require OPENAI_API_KEY to be set.
"""
