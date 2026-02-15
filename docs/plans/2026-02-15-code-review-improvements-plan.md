# Code Review Improvements Plan (v2)

**Date:** 2026-02-15
**Context:** Multi-persona code review (CTO, Principal Engineer, CPO) + peer review of plan
**Goal:** Address highest-impact feedback before submission

---

## Key Insight from Plan Review

> "The biggest miss is that the plan optimizes for polish rather than capability. One 'accept this ambiguity and regenerate' interaction would land harder than formatted emoji tables."

**Revised priority:** Capability > Polish

---

## Summary of Review Findings

| Dimension | Grade | Key Issue |
|-----------|-------|-----------|
| Code Quality | B+ | Solid patterns, good tests, but global state complexity and thin error messages |
| Architecture | B | Correct but over-engineered for a demo (circuit breaker, rate limiter for single-user CLI?) |
| Testing | A- | 243 tests, good coverage, but evals test selection not quality |
| Documentation | A | Excellent README, thorough AI attribution, clear setup instructions |
| Security | B- | Path traversal handled, but prompt injection disclaimed rather than mitigated |
| Agent Quality | C+ | 119 lines, no error handling, no state, no retry—demo tier |
| Product Polish | B | Value exists but buried; no "wow" moment in demo flow |
| Scope Management | B | Time accounting unclear given 9 AI sessions vs 6-hour constraint |

---

## Revised Priority Order

Based on peer feedback, the new priority order is:

1. **PR 2: Output Quality Evals** — non-negotiable, this is the credibility gap
2. **QW3 + Terminal Recording** — biggest ROI for reviewer experience
3. **PR 3: Formatted Output** — makes the demo scannable
4. **PR 1a: Agent Reliability** — stops the demo from looking fragile
5. **PR 1b: Ambiguity Feedback Loop** — the differentiator ⭐ NEW
6. Everything else is gravy

---

## PR Plan

### PR 2: Output Quality Evals 🔴 CRITICAL (Do First)

**Effort:** ~60 min
**Impact:** Non-negotiable credibility gap
**Branch:** `feat/output-quality-evals`

Current evals only verify tool selection. This PR adds evals that test actual output quality.

**Files:**
- `evals/eval_output_quality.py` (new)

**Tasks:**
- [ ] Add eval: "Given PRD with 'fast' and 'scalable', ambiguity_flags contains vague_quantifier"
- [ ] Add eval: "Given PRD with clear acceptance criteria, no critical ambiguity flags"
- [ ] Add eval: "Stories have valid requirement_ids referencing input requirements"
- [ ] Add eval: "Epic count is reasonable (1-4 for typical PRD)"
- [ ] Add eval: "All stories have size S/M/L assigned"

**Example eval structure:**
```python
@tool_eval()
async def output_quality_eval_suite() -> EvalSuite:
    suite = EvalSuite(name="PRD Decomposer Output Quality")

    # Eval: Vague quantifiers are flagged
    suite.add_case(
        name="Flags vague quantifiers",
        user_message="Analyze this PRD:\n\nThe system should be fast and scalable.",
        expected_output_validator=lambda output: (
            any(flag["category"] == "vague_quantifier"
                for req in output.get("requirements", [])
                for flag in req.get("ambiguity_flags", []))
        ),
    )

    return suite
```

**Acceptance Criteria:**
- [ ] 5+ output quality evals (not just tool selection)
- [ ] At least one eval for ambiguity detection accuracy
- [ ] At least one eval for ticket structure validation
- [ ] All evals pass with current implementation

**Test Commands:**
```bash
uv run arcade evals evals/eval_output_quality.py
```

---

### PR 1a: Agent Reliability 🔴 HIGH PRIORITY

**Effort:** ~30 min
**Impact:** Stops demo from looking fragile
**Branch:** `fix/agent-reliability`

Focus on reliability only (not interactivity—that's PR 1b).

**Files:**
- `agent/agent.py`

**Tasks:**
- [ ] Replace bare `except Exception` with specific error handling + traceback
- [ ] Add graceful timeout handling with user feedback ("LLM is taking longer than expected...")
- [ ] Add retry logic for MCP server connection failures (3x with backoff)
- [ ] Add `--verbose` flag for debugging

**Acceptance Criteria:**
- [ ] Errors show actionable messages, not raw exceptions
- [ ] Connection failures retry 3x with backoff
- [ ] Timeout shows user-friendly message

**Test Commands:**
```bash
uv run python agent/agent.py
uv run python agent/agent.py --verbose
```

---

### PR 1b: Ambiguity Feedback Loop 🔴 HIGH PRIORITY ⭐ NEW

**Effort:** ~45 min
**Impact:** The differentiator—transforms from tool to workflow
**Branch:** `feat/ambiguity-feedback`

> "Even a simple in-memory 'accept/dismiss ambiguity' would be transformative."

This is what makes the demo memorable. User can interact with results, not just view them.

**Files:**
- `agent/agent.py`
- `agent/session_state.py` (new)

**Concept:**
```
You: analyze samples/sample_prd.md

Agent: Found 5 requirements with 3 ambiguities:

  [1] 🔴 REQ-001: "fast" is undefined
  [2] 🟡 REQ-003: Missing error handling
  [3] 💡 REQ-002: "user-friendly" is vague

Commands: accept [n], dismiss [n], clarify [n] "text", tickets

You: accept 3
Agent: Accepted ambiguity #3. 2 remaining.

You: clarify 1 "Response time under 200ms p99"
Agent: Updated REQ-001 with clarification. Regenerating...

You: tickets
Agent: Generating tickets with resolved ambiguities...
```

**Tasks:**
- [ ] Create `session_state.py` with:
  ```python
  @dataclass
  class SessionState:
      current_requirements: dict | None = None
      accepted_ambiguities: set[str] = field(default_factory=set)
      clarifications: dict[str, str] = field(default_factory=dict)
  ```
- [ ] Add command parsing in agent loop: `accept`, `dismiss`, `clarify`, `tickets`
- [ ] Update agent instructions to explain commands
- [ ] Store analysis results in session state for modification
- [ ] Pass clarifications to decompose prompt

**Acceptance Criteria:**
- [ ] User can `accept` an ambiguity (removes from display, keeps in data)
- [ ] User can `clarify` an ambiguity with additional context
- [ ] Clarifications are passed to ticket decomposition
- [ ] State persists within session (resets on exit)

**Test Commands:**
```bash
uv run python agent/agent.py
# Then: analyze samples/sample_prd_01_rate_limiting.md
# Then: accept 1
# Then: clarify 2 "Must complete in under 3 steps"
# Then: tickets
```

---

### PR 3: UX Polish - Formatted Output 🟡 MEDIUM PRIORITY

**Effort:** ~45 min
**Impact:** Makes the demo scannable
**Branch:** `feat/formatted-output`

Make the agent output beautiful and scannable.

**Files:**
- `agent/agent.py`
- `agent/formatters.py` (new)

**Tasks:**
- [ ] Create `formatters.py` with helper functions:
  - `format_requirements_table()` - markdown table of requirements
  - `format_ambiguity_summary()` - numbered list with severity
  - `format_tickets_hierarchy()` - epic → story tree view
- [ ] Update agent instructions to request formatted output
- [ ] Add "⚠️ N ambiguities to resolve" summary at top of analysis

**Example output format:**
```
## ⚠️ 3 Ambiguities to Resolve

  [1] 🔴 critical | REQ-001 | "fast" undefined
      → Define latency SLA (e.g., <200ms p99)

  [2] 🟡 warning | REQ-003 | Missing error handling
      → Add error scenarios to acceptance criteria

  [3] 💡 suggestion | REQ-002 | "user-friendly" vague
      → Define measurable UX criteria

Commands: accept [n], dismiss [n], clarify [n] "text", tickets

## Requirements (5)

| ID | Title | Priority | Ambiguities |
|----|-------|----------|-------------|
| REQ-001 | User authentication | high | 1 |
| REQ-002 | Dashboard UI | medium | 1 |
| REQ-003 | Error handling | medium | 1 |
...
```

**Acceptance Criteria:**
- [ ] Ambiguity summary appears first with numbered list
- [ ] Commands hint shown after ambiguities
- [ ] Severity uses emoji indicators (🔴 🟡 💡)
- [ ] Ticket hierarchy shows epic → story relationship

---

### PR 4: Cost Transparency 🟢 LOWER PRIORITY

**Effort:** ~30 min
**Impact:** Addresses CTO cost concern
**Branch:** `feat/cost-transparency`

Surface token usage as estimated cost to the user.

**Files:**
- `src/prd_decomposer/config.py`
- `src/prd_decomposer/server.py`
- `agent/agent.py`
- `tests/test_config.py`

**Tasks:**
- [ ] Add cost settings to `Settings`:
  ```python
  cost_per_1k_input_tokens: float = Field(default=0.0025)
  cost_per_1k_output_tokens: float = Field(default=0.01)
  ```
- [ ] Add `estimated_cost_usd` to `_metadata` in tool responses
- [ ] Update agent to show cost after each operation
- [ ] Add cumulative session cost tracking in agent

**Acceptance Criteria:**
- [ ] Each tool response includes `estimated_cost_usd` in metadata
- [ ] Agent shows cost after each operation: "Analysis complete ($0.03)"
- [ ] Session total shown on exit

---

### PR 5: Single-Command Demo Flow 🟢 LOWER PRIORITY

**Effort:** ~40 min
**Impact:** Convenience (but PR 1b is more important)
**Branch:** `feat/single-command-flow`

**Note:** Peer review flagged this as "tangential"—it adds convenience but doesn't address iterative workflow. Deprioritized in favor of PR 1b.

**Files:**
- `src/prd_decomposer/server.py`
- `tests/test_server.py`

**Tasks:**
- [ ] Add `analyze_and_decompose` tool to server.py
- [ ] Add unit tests for combined flow
- [ ] Update agent instructions to mention combined tool

---

### PR 6: Prompt Injection Mitigations 🟢 LOWER PRIORITY

**Effort:** ~45 min
**Impact:** Security posture improvement
**Branch:** `feat/output-validation`

**Note:** Peer review noted "the README already disclaims it honestly, which is the right posture for a demo." Only do if time permits.

**Files:**
- `src/prd_decomposer/validators.py` (new)
- `tests/test_validators.py` (new)

---

## Quick Wins

### QW3: Demo Script + Terminal Recording 🔴 HIGH PRIORITY

**Effort:** ~20 min
**Impact:** Biggest ROI for reviewer experience

**Tasks:**
- [ ] Add "Quick Demo" section to README with 5-step narrative
- [ ] Record terminal session with asciinema or create GIF
- [ ] Embed recording in README

**Demo script:**
```markdown
## Quick Demo

1. **Start with a vague PRD** - `samples/sample_prd_01_rate_limiting.md`
2. **Analyze** - See 5 ambiguities flagged automatically
3. **Resolve** - `accept 1`, `clarify 2 "under 200ms"`
4. **Decompose** - Get 12 stories in 8 seconds
5. **Trace** - Every story links to source requirements

![Demo](docs/demo.gif)
```

**Recording commands:**
```bash
# Option 1: asciinema (uploads to asciinema.org)
asciinema rec demo.cast
# run demo
asciinema upload demo.cast

# Option 2: terminalizer (local GIF)
terminalizer record demo
terminalizer render demo -o docs/demo.gif
```

---

### QW2: Cost Estimate in README

**Effort:** ~5 min

Add to README under "Features":
```markdown
### Cost Efficiency
- Typical PRD analysis: ~$0.02-0.05
- Ticket decomposition: ~$0.03-0.08
- Full workflow: ~$0.05-0.15 per PRD
- Process 50 PRDs/month for under $10
```

---

### QW1: Before/After Showcase

**Effort:** ~15 min

Create `docs/showcase.md` with side-by-side comparison.

---

## Gaps Acknowledged (Not Addressing)

Per peer review, these items were flagged but won't be addressed due to time:

| Gap | Reason for Skipping |
|-----|---------------------|
| `_call_llm_with_retry` complexity | Works correctly; refactor is polish |
| mypy not in CI | Definition of Done covers it; CI is nice-to-have |
| RateLimitError/APIError ordering | Minor; comment would help but not critical |
| project_key unvalidated | Minor edge case |

---

## Revised Schedule

| Time Block | Item | Duration |
|------------|------|----------|
| 9:00-10:00 | **PR 2: Output Quality Evals** | 60 min |
| 10:00-10:20 | QW3: Demo script (text part) | 20 min |
| 10:20-10:30 | Break | 10 min |
| 10:30-11:00 | **PR 1a: Agent Reliability** | 30 min |
| 11:00-11:45 | **PR 1b: Ambiguity Feedback Loop** ⭐ | 45 min |
| 11:45-12:00 | QW2: Cost estimate in README | 15 min |
| 12:00-13:00 | Lunch | — |
| 13:00-13:45 | **PR 3: Formatted Output** | 45 min |
| 13:45-14:15 | QW3: Terminal recording | 30 min |
| 14:15-14:45 | PR 4: Cost Transparency (if time) | 30 min |
| 14:45-15:00 | Final review + commit | 15 min |

**Total productive time:** ~5.5 hours

---

## Git Workflow

```bash
# For each PR
git checkout main
git pull
git checkout -b <branch-name>

# Make changes, then
uv run pytest tests/ -v
uv run ruff check src/ tests/ agent/
uv run mypy src/

# Commit
git add -A
git commit -m "feat: <description>"
git push -u origin <branch-name>

# Merge to main (or create PR)
git checkout main
git merge <branch-name>
git push
```

---

## Definition of Done

Each PR is complete when:
- [ ] All new code has tests (where applicable)
- [ ] `uv run pytest tests/ -v` passes
- [ ] `uv run ruff check src/ tests/` passes
- [ ] `uv run mypy src/` passes
- [ ] README updated if user-facing change
- [ ] AI_USAGE.md updated with session notes

---

## Success Criteria for Submission

The demo should show:
1. ✅ Vague PRD → ambiguities flagged (proves the "why")
2. ✅ User can interact with ambiguities (proves the workflow)
3. ✅ Clean tickets generated (proves the value)
4. ✅ Output quality validated by evals (proves reliability)
5. ✅ Doesn't crash during demo (proves production-readiness)
