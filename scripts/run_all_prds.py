#!/usr/bin/env python3
"""Batch process all sample PRDs and save results.

Run with: uv run python scripts/run_all_prds.py
"""

import json
import sys
from pathlib import Path

from prd_decomposer.server import analyze_prd, decompose_to_tickets, read_file


def process_prd(prd_path: Path, output_dir: Path) -> dict:
    """Process a single PRD and return results."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {prd_path.name}")
    print("=" * 60)

    # Read the PRD
    prd_text = read_file(str(prd_path))
    print(f"  Read {len(prd_text)} characters")

    # Analyze
    print("  Analyzing PRD...")
    requirements = analyze_prd(prd_text)
    print(f"  Found {len(requirements['requirements'])} requirements")

    # Count ambiguities
    ambiguity_count = sum(
        len(req.get("ambiguity_flags", [])) for req in requirements["requirements"]
    )
    print(f"  Ambiguity flags: {ambiguity_count}")

    # Decompose to tickets
    print("  Generating Jira tickets...")
    tickets = decompose_to_tickets(requirements)
    epic_count = len(tickets["epics"])
    story_count = tickets["metadata"]["story_count"]
    print(f"  Generated {epic_count} epics with {story_count} stories")

    # Save outputs
    stem = prd_path.stem

    # Save requirements
    req_file = output_dir / f"{stem}_requirements.json"
    req_file.write_text(json.dumps(requirements, indent=2))
    print(f"  Saved: {req_file.name}")

    # Save tickets
    tickets_file = output_dir / f"{stem}_tickets.json"
    tickets_file.write_text(json.dumps(tickets, indent=2))
    print(f"  Saved: {tickets_file.name}")

    return {
        "prd": prd_path.name,
        "requirements_count": len(requirements["requirements"]),
        "ambiguity_count": ambiguity_count,
        "epic_count": epic_count,
        "story_count": story_count,
    }


def main() -> None:
    """Process all sample PRDs."""
    samples_dir = Path(__file__).parent.parent / "samples"
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Find all sample PRDs
    prd_files = sorted(samples_dir.glob("sample_prd_*.md"))

    if not prd_files:
        print("No sample PRD files found!")
        sys.exit(1)

    print(f"Found {len(prd_files)} sample PRDs")
    print(f"Output directory: {output_dir}")

    results = []
    for prd_path in prd_files:
        try:
            result = process_prd(prd_path, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"prd": prd_path.name, "error": str(e)})

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'PRD':<45} {'Reqs':>5} {'Ambig':>6} {'Epics':>6} {'Stories':>8}")
    print("-" * 60)

    total_reqs = 0
    total_ambig = 0
    total_epics = 0
    total_stories = 0

    for r in results:
        if "error" in r:
            print(f"{r['prd']:<45} ERROR: {r['error']}")
        else:
            print(
                f"{r['prd']:<45} {r['requirements_count']:>5} {r['ambiguity_count']:>6} {r['epic_count']:>6} {r['story_count']:>8}"
            )
            total_reqs += r["requirements_count"]
            total_ambig += r["ambiguity_count"]
            total_epics += r["epic_count"]
            total_stories += r["story_count"]

    print("-" * 60)
    print(f"{'TOTAL':<45} {total_reqs:>5} {total_ambig:>6} {total_epics:>6} {total_stories:>8}")

    # Save summary
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
