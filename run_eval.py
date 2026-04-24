"""
Eval runner: runs all harness × model combinations with multiple trials.

Matrix:
- 3 harnesses (A: implicit, B: explicit, C: adaptive)
- 2 models (claude-sonnet-4-20250514, gpt-4o)
- N trials per combination (default: 5)

Outputs:
- Summary table with completion rate, avg iterations, false finishes, etc.
- Detailed per-run logs saved to JSON
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Initialize tracing BEFORE importing harnesses (which import models)
from instrumentation import init_tracing, shutdown_tracing

import harness_a
import harness_b
import harness_c
from models import EVAL_MODELS, Provider

console = Console()

# Available harnesses
HARNESSES = {
    "a": ("implicit", harness_a),
    "b": ("explicit", harness_b),
    "c": ("adaptive", harness_c),
}

# Available providers/models for eval matrix
EVAL_MATRIX = [
    ("anthropic", EVAL_MODELS["anthropic"]),
    ("openai", EVAL_MODELS["openai"]),
]

# Default test prompt - complex enough to exercise the tools
DEFAULT_EVAL_PROMPT = """Find all spans containing 'Aragorn' in the trace.
For each one, get the full span data and extract the latency.
Sort by latency (highest first) and multiply each latency by 314.
Present the results in a markdown table with columns: span_id, original_latency, multiplied_latency."""


@dataclass
class RunResult:
    """Result from a single eval run."""
    harness: str
    harness_label: str
    provider: str
    model: str
    trial: int
    prompt: str

    # Core metrics
    task_complete: bool
    iterations: int
    tool_calls: int
    false_finishes: int
    narrate_then_act: int
    total_tokens: int
    input_tokens: int
    output_tokens: int

    # Additional context
    finished_reason: str
    text: str
    error: str | None = None


def run_single(
    harness_name: str,
    provider: Provider,
    model: str,
    prompt: str,
    trial: int,
) -> RunResult:
    """Run a single harness trial and return structured result."""
    harness_label, harness_module = HARNESSES[harness_name]

    try:
        result = harness_module.run(
            prompt=prompt,
            provider=provider,
            model=model,
        )

        return RunResult(
            harness=harness_name,
            harness_label=harness_label,
            provider=provider,
            model=model,
            trial=trial,
            prompt=prompt,
            task_complete=result.get("task_complete", False),
            iterations=result.get("iterations", 0),
            tool_calls=result.get("tool_calls_total", 0),
            false_finishes=result.get("false_finishes", 0),
            narrate_then_act=result.get("narrate_then_act", 0),
            total_tokens=result.get("total_tokens", 0),
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            finished_reason=result.get("finished_reason", "unknown"),
            text=result.get("text", ""),
        )

    except Exception as e:
        console.print(f"[red]ERROR in {harness_label}/{provider} trial {trial}: {e}[/red]")
        return RunResult(
            harness=harness_name,
            harness_label=harness_label,
            provider=provider,
            model=model,
            trial=trial,
            prompt=prompt,
            task_complete=False,
            iterations=0,
            tool_calls=0,
            false_finishes=0,
            narrate_then_act=0,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            finished_reason="error",
            text="",
            error=str(e),
        )


def run_eval_matrix(
    prompt: str,
    trials: int = 5,
    harnesses: list[str] | None = None,
    providers: list[str] | None = None,
) -> list[RunResult]:
    """
    Run the full evaluation matrix.

    Args:
        prompt: The eval prompt to use
        trials: Number of trials per combination
        harnesses: List of harness names to test (default: all)
        providers: List of providers to test (default: all)

    Returns:
        List of RunResult objects
    """
    harnesses = harnesses or list(HARNESSES.keys())

    # Filter EVAL_MATRIX by providers if specified
    matrix = EVAL_MATRIX
    if providers:
        matrix = [(p, m) for p, m in EVAL_MATRIX if p in providers]

    results: list[RunResult] = []
    total_runs = len(harnesses) * len(matrix) * trials
    current_run = 0

    console.print(Panel(
        f"[bold]Harnesses:[/bold] {', '.join(harnesses)}\n"
        f"[bold]Models:[/bold] {', '.join(f'{p}/{m}' for p, m in matrix)}\n"
        f"[bold]Trials per combo:[/bold] {trials}\n"
        f"[bold]Total runs:[/bold] {total_runs}",
        title="[bold cyan]Eval Matrix[/bold cyan]",
        border_style="cyan",
    ))

    for harness_name in harnesses:
        harness_label, _ = HARNESSES[harness_name]

        for provider, model in matrix:
            for trial in range(1, trials + 1):
                current_run += 1

                console.print(f"\n[bold]═══ Run {current_run}/{total_runs}: {harness_label} / {provider} / Trial {trial} ═══[/bold]")

                result = run_single(
                    harness_name=harness_name,
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    trial=trial,
                )
                results.append(result)

                # Quick status
                status = "[green]✓[/green]" if result.task_complete else "[red]✗[/red]"
                console.print(f"  {status} iterations={result.iterations}, tokens={result.total_tokens:,}, false_finishes={result.false_finishes}")

    return results


def compute_summary(results: list[RunResult]) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Compute summary statistics grouped by (harness, provider).

    Returns:
        Dict mapping (harness_label, provider) to summary stats
    """
    from collections import defaultdict

    groups: dict[tuple[str, str], list[RunResult]] = defaultdict(list)

    for r in results:
        key = (r.harness_label, r.provider)
        groups[key].append(r)

    summaries = {}
    for key, group in groups.items():
        n = len(group)
        completed = sum(1 for r in group if r.task_complete)

        summaries[key] = {
            "trials": n,
            "completed": completed,
            "completion_rate": f"{completed}/{n}",
            "avg_iterations": sum(r.iterations for r in group) / n if n > 0 else 0,
            "total_false_finishes": sum(r.false_finishes for r in group),
            "total_narrate_then_act": sum(r.narrate_then_act for r in group),
            "avg_tokens": sum(r.total_tokens for r in group) / n if n > 0 else 0,
            "total_tokens": sum(r.total_tokens for r in group),
        }

    return summaries


def print_summary_table(summaries: dict[tuple[str, str], dict[str, Any]]):
    """Print a rich summary table."""
    table = Table(title="Eval Results Summary", show_header=True, header_style="bold cyan")

    table.add_column("Harness", style="bold")
    table.add_column("Model", style="dim")
    table.add_column("Completion", justify="center")
    table.add_column("Avg Turns", justify="right")
    table.add_column("False Finishes", justify="right")
    table.add_column("Narrate-then-Act", justify="right")
    table.add_column("Avg Tokens", justify="right")

    # Sort by harness then provider for consistent display
    for (harness_label, provider), stats in sorted(summaries.items()):
        # Color completion rate based on success
        completed, total = stats["completion_rate"].split("/")
        if int(completed) == int(total):
            completion_str = f"[green]{stats['completion_rate']}[/green]"
        elif int(completed) == 0:
            completion_str = f"[red]{stats['completion_rate']}[/red]"
        else:
            completion_str = f"[yellow]{stats['completion_rate']}[/yellow]"

        table.add_row(
            harness_label,
            provider,
            completion_str,
            f"{stats['avg_iterations']:.1f}",
            str(stats['total_false_finishes']),
            str(stats['total_narrate_then_act']),
            f"{stats['avg_tokens']:,.0f}",
        )

    console.print()
    console.print(table)


def save_results(
    results: list[RunResult],
    summaries: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
    prompt: str,
):
    """Save detailed results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "summary": {
            f"{h}_{p}": stats
            for (h, p), stats in summaries.items()
        },
        "runs": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Run finish-condition eval matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py                          # Run all combos, 5 trials each
  python run_eval.py -t 10                    # 10 trials per combo
  python run_eval.py -H a -H b -p anthropic   # Only harnesses A/B with Claude
  python run_eval.py -o results.json          # Save detailed results
        """,
    )
    parser.add_argument(
        "--harness", "-H",
        choices=list(HARNESSES.keys()),
        action="append",
        help="Harness(es) to run (default: all)",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "openai"],
        action="append",
        help="Provider(s) to use (default: all)",
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=5,
        help="Number of trials per combination (default: 5)",
    )
    parser.add_argument(
        "--prompt", "-P",
        default=DEFAULT_EVAL_PROMPT,
        help="Eval prompt to use",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for detailed results (JSON)",
    )
    parser.add_argument(
        "--project",
        default="harness-evals",
        help="Phoenix project name (default: harness-evals)",
    )

    args = parser.parse_args()

    # Initialize tracing
    init_tracing(project_name=args.project)

    try:
        # Run the eval matrix
        results = run_eval_matrix(
            prompt=args.prompt,
            trials=args.trials,
            harnesses=args.harness,
            providers=args.provider,
        )

        # Compute and display summary
        summaries = compute_summary(results)
        print_summary_table(summaries)

        # Save results if requested
        if args.output:
            save_results(results, summaries, args.output, args.prompt)

    finally:
        # Always flush traces
        shutdown_tracing()


if __name__ == "__main__":
    main()
