"""
Evaluation runner with Phoenix tracing.

This module wraps the clean harness functions with observability.
Tracing is decoupled from harness logic - harnesses stay readable.

Usage:
    # Run single harness with tracing
    python evals.py --harness C --provider anthropic

    # Run all harnesses
    python evals.py --all

    # Run with GAIA tools
    python evals.py --harness C --gaia
"""

import argparse
import json
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table

import harness_implicit
import harness_explicit
import harness_adaptive

console = Console()

HARNESSES = {
    "implicit": ("implicit", harness_implicit.run),
    "explicit": ("explicit", harness_explicit.run),
    "adaptive": ("adaptive", harness_adaptive.run),
}


# ─────────────────────────────────────────────────────────────────────────────
# Tracing setup
# ─────────────────────────────────────────────────────────────────────────────

_tracer = None
_tracing_initialized = False


def init_tracing(project_name: str = "harness-evals"):
    """Initialize Phoenix tracing. Safe to call multiple times."""
    global _tracing_initialized, _tracer

    if _tracing_initialized:
        return

    try:
        from phoenix.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace

        register(project_name=project_name, endpoint="http://localhost:6006/v1/traces")
        AnthropicInstrumentor().instrument()
        OpenAIInstrumentor().instrument()

        _tracer = trace.get_tracer("harness-evals")
        _tracing_initialized = True

        console.print(f"[dim]Tracing enabled → http://localhost:6006[/dim]")

    except ImportError as e:
        console.print(f"[yellow]Tracing disabled (missing deps): {e}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Tracing disabled: {e}[/yellow]")


def shutdown_tracing():
    """Flush and shutdown tracing."""
    if not _tracing_initialized:
        return

    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, 'force_flush'):
            provider.force_flush()
        if hasattr(provider, 'shutdown'):
            provider.shutdown()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Span helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _chain_span(name: str, **attributes):
    """Create a CHAIN span for the agent run."""
    if not _tracer:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span


@contextmanager
def _iteration_span(n: int, **attributes):
    """Create a span for a single iteration."""
    if not _tracer:
        yield None
        return

    with _tracer.start_as_current_span(f"iteration_{n}") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("iteration_number", n)
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span


@contextmanager
def _tool_span(name: str, arguments: dict, result: str):
    """Create a TOOL span."""
    if not _tracer:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("input.value", json.dumps(arguments))
        span.set_attribute("output.value", result[:1000])  # Truncate
        yield span


# ─────────────────────────────────────────────────────────────────────────────
# Main eval functions
# ─────────────────────────────────────────────────────────────────────────────

def run_with_tracing(
    harness_name: str,
    prompt: str,
    provider: str = "anthropic",
    model: str | None = None,
    tools_list: list | None = None,
    tool_functions: dict | None = None,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Run a harness with Phoenix tracing.

    Wraps the clean harness and creates spans from the returned iteration data.
    """
    harness_type, harness_fn = HARNESSES[harness_name]

    # Run the harness (clean, no tracing inside)
    result = harness_fn(
        prompt=prompt,
        provider=provider,
        model=model,
        tools_list=tools_list,
        tool_functions=tool_functions,
        verbose=verbose,
        **kwargs,
    )

    # Create spans from the result data (if tracing enabled)
    with _chain_span(
        "agent_run",
        harness_type=harness_type,
        model_provider=provider,
        model_name=model or "default",
    ) as chain:
        if chain:
            chain.set_attribute("input.value", prompt)

        # Create iteration spans from recorded data
        for iter_data in result.get("iterations", []):
            with _iteration_span(
                iter_data["n"],
                finish_triggered=iter_data.get("finish_triggered", False),
                false_finish=iter_data.get("false_finish", False),
                narrate_then_act=iter_data.get("narrate_then_act", False),
            ):
                # Create tool spans
                for tool in iter_data.get("tool_calls", []):
                    with _tool_span(tool["name"], tool["arguments"], tool["result"]):
                        pass

        if chain:
            chain.set_attribute("output.value", result.get("text", "")[:2000])
            chain.set_attribute("task_complete", result.get("task_complete", False))
            chain.set_attribute("finished_reason", result.get("finished_reason", "unknown"))
            chain.set_attribute("false_finishes", result.get("false_finishes", 0))

    result["harness"] = harness_name
    result["provider"] = provider
    return result


def run_eval_matrix(
    prompts: list[str],
    harnesses: list[str] = ["implicit", "explicit", "adaptive"],
    providers: list[str] = ["anthropic"],
    tools_list: list | None = None,
    tool_functions: dict | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Run evaluation across harnesses and providers."""
    results = []

    total = len(prompts) * len(harnesses) * len(providers)
    run_num = 0

    for prompt in prompts:
        for provider in providers:
            for harness in harnesses:
                run_num += 1
                console.print(f"\n[cyan]Run {run_num}/{total}:[/cyan] Harness {harness} | {provider}")

                result = run_with_tracing(
                    harness_name=harness,
                    prompt=prompt,
                    provider=provider,
                    tools_list=tools_list,
                    tool_functions=tool_functions,
                    verbose=verbose,
                )
                results.append(result)

                # Quick summary
                status = "[green]Complete[/green]" if result["task_complete"] else "[red]Incomplete[/red]"
                console.print(
                    f"   {status} | {result['iteration_count']} iters | "
                    f"{result['tool_calls_total']} tools | {result['total_tokens']:,} tokens"
                )

    return results


def print_results_table(results: list[dict]) -> None:
    """Print summary table of results."""
    table = Table(title="Eval Results")

    table.add_column("Harness", style="cyan")
    table.add_column("Provider")
    table.add_column("Complete", justify="center")
    table.add_column("Iters", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("False Fin", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Finished")

    for r in results:
        complete = "[green]Yes[/green]" if r["task_complete"] else "[red]No[/red]"
        table.add_row(
            r["harness"],
            r["provider"],
            complete,
            str(r["iteration_count"]),
            str(r["tool_calls_total"]),
            str(r.get("false_finishes", 0)),
            f"{r['total_tokens']:,}",
            r["finished_reason"][:15],
        )

    console.print(table)


def print_summary_by_harness(results: list[dict]) -> None:
    """Print aggregate summary by harness."""
    by_harness: dict[str, list[dict]] = {}
    for r in results:
        h = r["harness"]
        if h not in by_harness:
            by_harness[h] = []
        by_harness[h].append(r)

    table = Table(title="Summary by Harness")
    table.add_column("Harness")
    table.add_column("Runs", justify="right")
    table.add_column("Completed", justify="right")
    table.add_column("Avg Iters", justify="right")
    table.add_column("Avg Tools", justify="right")
    table.add_column("Total False Fin", justify="right")
    table.add_column("Total Tokens", justify="right")

    for harness in sorted(by_harness.keys()):
        runs = by_harness[harness]
        n = len(runs)
        completed = sum(1 for r in runs if r["task_complete"])
        avg_iters = sum(r["iteration_count"] for r in runs) / n
        avg_tools = sum(r["tool_calls_total"] for r in runs) / n
        total_ff = sum(r.get("false_finishes", 0) for r in runs)
        total_tokens = sum(r["total_tokens"] for r in runs)

        table.add_row(
            f"Harness {harness}",
            str(n),
            f"{completed}/{n} ({100*completed/n:.0f}%)",
            f"{avg_iters:.1f}",
            f"{avg_tools:.1f}",
            str(total_ff),
            f"{total_tokens:,}",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Run harness evaluations with tracing")

    parser.add_argument(
        "--harness", "-H",
        choices=["implicit", "explicit", "adaptive"],
        help="Run specific harness (default: all)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all harnesses",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "openai", "both"],
        default="anthropic",
        help="Model provider (use 'both' for full matrix)",
    )
    parser.add_argument(
        "--prompt", "-P",
        default="Can you find the populations of Tokyo, Osaka, and Yokohama, sort them by size, multiply each by 314, and show me the results in a Markdown table?",
        help="Prompt to test",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output during runs",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable tracing",
    )

    args = parser.parse_args()

    # Initialize tracing
    if not args.no_trace:
        init_tracing()

    # Load tools (web search, calculator, etc.)
    from tools import tools_list, tool_functions

    # Determine which harnesses to run
    if args.all or not args.harness:
        harnesses = ["implicit", "explicit", "adaptive"]
    else:
        harnesses = [args.harness]

    # Determine which providers to run
    if args.provider == "both":
        providers = ["anthropic", "openai"]
    else:
        providers = [args.provider]

    # Run eval
    results = run_eval_matrix(
        prompts=[args.prompt],
        harnesses=harnesses,
        providers=providers,
        tools_list=tools_list,
        tool_functions=tool_functions,
        verbose=args.verbose,
    )

    # Print results
    console.print()
    print_results_table(results)
    if len(harnesses) > 1:
        print_summary_by_harness(results)

    # Shutdown tracing
    shutdown_tracing()


if __name__ == "__main__":
    main()
