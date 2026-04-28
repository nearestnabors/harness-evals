"""
Run multi-step benchmark tasks across harnesses.

Usage:
    # Run sample tasks
    python run_benchmark.py --sample

    # Run specific harness only
    python run_benchmark.py --sample --harness implicit

    # Run with both providers
    python run_benchmark.py --sample --provider both
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

from instrumentation import init_tracing, shutdown_tracing
from eval_tasks import (
    EvalTask,
    get_sample_tasks,
    format_task_prompt,
    check_answer,
)
from tools import tools_list, tool_functions, reset_tasks
from models import EVAL_MODELS

console = Console()


# System prompt for benchmark tasks designed to encourage acknowledgement + planning
SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
You are a polite, conversational agent.
- ALWAYS start with a short acknowledgement to the user
- THEN, if a tool is needed, include the tool call
- Keep the user informed of what you're doing

IMPORTANT: Plan and track your tasks throughout the conversation.
1. First acknowledge the request
2. Create a plan for what you need to do
3. Execute each step one at a time
4. Report results as you go

It is critical that you complete each step before moving to the next.
Respond politely and conversationally, starting with a brief acknowledgement.

Available tools:
- web_search: Search the web for current information
- wikipedia_lookup: Look up facts on Wikipedia
- calculator: Perform mathematical calculations"""


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""

    task_id: str
    level: int
    harness: str
    provider: str
    model: str
    task_complete: bool
    answer_correct: bool | None  # None if no expected answer
    iterations: int
    tool_calls: int
    false_finishes: int
    narrate_then_act: int
    total_tokens: int
    finished_reason: str
    model_answer: str
    expected_answer: str | None
    error: str | None = None


def run_task_with_harness(
    task: EvalTask,
    harness: str,
    provider: str,
    model: str,
) -> TaskResult:
    """Run a single benchmark task with a specific harness."""

    prompt = format_task_prompt(task, include_instructions=False)

    # Import the appropriate harness
    if harness == "implicit":
        from harness_implicit import run as run_harness
    elif harness == "explicit":
        from harness_explicit import run as run_harness
    elif harness == "adaptive":
        from harness_adaptive import run as run_harness
    else:
        raise ValueError(f"Unknown harness: {harness}")

    try:
        # Reset tools state
        reset_tasks()

        # Run the harness
        result = run_harness(
            prompt=prompt,
            provider=provider,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            max_iterations=15,
            tools_list=tools_list,
            tool_functions=tool_functions,
            verbose=True,
        )

        # Check if answer is correct
        model_answer = result.get("text", "")
        answer_correct = None
        if task.expected_answer:
            answer_correct = check_answer(task, model_answer)

        return TaskResult(
            task_id=task.task_id,
            level=task.level,
            harness=harness,
            provider=provider,
            model=model,
            task_complete=result.get("task_complete", False),
            answer_correct=answer_correct,
            iterations=result.get("iteration_count", 0),
            tool_calls=result.get("tool_calls_total", 0),
            false_finishes=result.get("false_finishes", 0),
            narrate_then_act=result.get("narrate_then_act", 0),
            total_tokens=result.get("total_tokens", 0),
            finished_reason=result.get("finished_reason", "unknown"),
            model_answer=model_answer[:500],  # Truncate for storage
            expected_answer=task.expected_answer,
            error=None,
        )

    except Exception as e:
        return TaskResult(
            task_id=task.task_id,
            level=task.level,
            harness=harness,
            provider=provider,
            model=model,
            task_complete=False,
            answer_correct=False,
            iterations=0,
            tool_calls=0,
            false_finishes=0,
            narrate_then_act=0,
            total_tokens=0,
            finished_reason="error",
            model_answer="",
            expected_answer=task.expected_answer,
            error=str(e),
        )


def print_results_table(results: list[TaskResult]) -> None:
    """Print a summary table of results."""

    table = Table(title="Benchmark Results", show_lines=True)

    table.add_column("Task", style="cyan", width=12)
    table.add_column("Level", justify="center", width=5)
    table.add_column("Harness", justify="center", width=8)
    table.add_column("Provider", width=10)
    table.add_column("Complete", justify="center", width=8)
    table.add_column("Correct", justify="center", width=8)
    table.add_column("Iters", justify="right", width=5)
    table.add_column("Tools", justify="right", width=5)
    table.add_column("False Fin", justify="right", width=9)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Finished", width=15)

    for r in results:
        complete_str = "[green]Yes[/green]" if r.task_complete else "[red]No[/red]"

        if r.answer_correct is None:
            correct_str = "[dim]N/A[/dim]"
        elif r.answer_correct:
            correct_str = "[green]Yes[/green]"
        else:
            correct_str = "[red]No[/red]"

        if r.error:
            correct_str = f"[red]Err[/red]"

        table.add_row(
            r.task_id[:12],
            str(r.level),
            r.harness,
            r.provider,
            complete_str,
            correct_str,
            str(r.iterations),
            str(r.tool_calls),
            str(r.false_finishes),
            f"{r.total_tokens:,}",
            r.finished_reason[:15],
        )

    console.print(table)


def print_summary(results: list[TaskResult]) -> None:
    """Print aggregate summary statistics."""

    # Group by harness
    by_harness: dict[str, list[TaskResult]] = {}
    for r in results:
        if r.harness not in by_harness:
            by_harness[r.harness] = []
        by_harness[r.harness].append(r)

    console.print(Rule("Summary by Harness"))

    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Harness")
    summary_table.add_column("Tasks", justify="right")
    summary_table.add_column("Completed", justify="right")
    summary_table.add_column("Correct", justify="right")
    summary_table.add_column("Avg Iters", justify="right")
    summary_table.add_column("Avg Tools", justify="right")
    summary_table.add_column("Total False Fin", justify="right")
    summary_table.add_column("Total Tokens", justify="right")

    for harness in sorted(by_harness.keys()):
        hrs = by_harness[harness]
        n = len(hrs)
        completed = sum(1 for r in hrs if r.task_complete)
        correct = sum(1 for r in hrs if r.answer_correct is True)
        checkable = sum(1 for r in hrs if r.answer_correct is not None)
        avg_iters = sum(r.iterations for r in hrs) / n if n else 0
        avg_tools = sum(r.tool_calls for r in hrs) / n if n else 0
        total_ff = sum(r.false_finishes for r in hrs)
        total_tokens = sum(r.total_tokens for r in hrs)

        correct_str = f"{correct}/{checkable}" if checkable else "N/A"

        summary_table.add_row(
            f"Harness {harness}",
            str(n),
            f"{completed}/{n} ({100*completed/n:.0f}%)" if n else "0",
            correct_str,
            f"{avg_iters:.1f}",
            f"{avg_tools:.1f}",
            str(total_ff),
            f"{total_tokens:,}",
        )

    console.print(summary_table)


def main():
    parser = argparse.ArgumentParser(description="Run multi-step benchmark tasks")

    # Task selection
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to run (default: all)",
    )

    # Harness selection
    parser.add_argument(
        "--harness",
        choices=["implicit", "explicit", "adaptive", "all"],
        default="all",
        help="Which harness to run (default: all)",
    )

    # Provider selection
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "both"],
        default="anthropic",
        help="Which provider to use (default: anthropic)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Initialize tracing
    init_tracing()

    # Load tasks
    console.print(Rule("[bold]Loading Evaluation Tasks[/bold]"))
    tasks = get_sample_tasks()
    console.print(f"Loaded {len(tasks)} multi-step evaluation tasks")

    if args.max_tasks and len(tasks) > args.max_tasks:
        tasks = tasks[: args.max_tasks]

    # Determine harnesses to run
    if args.harness == "all":
        harnesses = ["implicit", "explicit", "adaptive"]
    else:
        harnesses = [args.harness]

    # Determine providers to run
    if args.provider == "both":
        providers = ["anthropic", "openai"]
    else:
        providers = [args.provider]

    # Show run configuration
    console.print(Panel(
        f"[bold]Tasks:[/bold] {len(tasks)}\n"
        f"[bold]Harnesses:[/bold] {', '.join(harnesses)}\n"
        f"[bold]Providers:[/bold] {', '.join(providers)}\n"
        f"[bold]Total runs:[/bold] {len(tasks) * len(harnesses) * len(providers)}",
        title="Run Configuration",
    ))

    # Run tasks
    results: list[TaskResult] = []
    total_runs = len(tasks) * len(harnesses) * len(providers)
    run_num = 0

    for task in tasks:
        console.print(Rule(f"[bold]Task: {task.task_id}[/bold] (Level {task.level})"))
        console.print(f"[dim]{task.question[:200]}...[/dim]\n" if len(task.question) > 200 else f"[dim]{task.question}[/dim]\n")

        for provider in providers:
            model = EVAL_MODELS[provider]

            for harness in harnesses:
                run_num += 1
                console.print(f"\n[cyan]Run {run_num}/{total_runs}:[/cyan] Harness {harness} | {provider} | {model}")

                result = run_task_with_harness(
                    task=task,
                    harness=harness,
                    provider=provider,
                    model=model,
                )
                results.append(result)

                # Quick result summary
                status = "[green]Complete[/green]" if result.task_complete else "[red]Incomplete[/red]"
                if result.answer_correct is True:
                    answer = "[green]Correct[/green]"
                elif result.answer_correct is False:
                    answer = "[red]Wrong[/red]"
                else:
                    answer = "[dim]N/A[/dim]"

                if result.error:
                    console.print(f"   [red]Error:[/red] {result.error}")
                else:
                    console.print(
                        f"   {status} | {answer} | "
                        f"{result.iterations} iters | {result.tool_calls} tools | "
                        f"{result.total_tokens:,} tokens"
                    )

    # Print results
    console.print("\n")
    console.print(Rule("[bold]Results[/bold]"))
    print_results_table(results)
    print_summary(results)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sample": args.sample,
                "level": args.level,
                "max_tasks": args.max_tasks,
                "harnesses": harnesses,
                "providers": providers,
            },
            "results": [asdict(r) for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output_path}[/green]")

    # Shutdown tracing
    shutdown_tracing()


if __name__ == "__main__":
    main()
