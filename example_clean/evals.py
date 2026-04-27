"""
Evaluation runner with Phoenix tracing.

Takes clean harness functions and adds observability.
Tracing is separate from harness logic - harnesses stay readable.
"""

import json
from contextlib import contextmanager
from opentelemetry import trace

# Import clean harnesses
import harness_implicit
import harness_explicit
import harness_adaptive

HARNESSES = {
    "implicit": harness_implicit.run,
    "explicit": harness_explicit.run,
    "adaptive": harness_adaptive.run,
}


def init_tracing(project_name: str = "harness-evals"):
    """Initialize Phoenix tracing."""
    from phoenix.otel import register
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    register(project_name=project_name, endpoint="http://localhost:6006/v1/traces")
    AnthropicInstrumentor().instrument()
    OpenAIInstrumentor().instrument()


def _get_tracer():
    return trace.get_tracer("harness-evals")


@contextmanager
def _chain_span(name: str, **attributes):
    """Create a CHAIN span for the agent run."""
    tracer = _get_tracer()
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        for k, v in attributes.items():
            span.set_attribute(k, v)
        yield span


@contextmanager
def _iteration_span(n: int, **attributes):
    """Create a span for a single iteration."""
    tracer = _get_tracer()
    with tracer.start_as_current_span(f"iteration_{n}") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("iteration_number", n)
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span


@contextmanager
def _tool_span(name: str, arguments: dict, result: str):
    """Create a TOOL span."""
    tracer = _get_tracer()
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("input.value", json.dumps(arguments))
        span.set_attribute("output.value", result)
        yield span


def run_with_tracing(
    harness_name: str,
    prompt: str,
    provider: str = "anthropic",
    model: str | None = None,
    tools_list: list | None = None,
    tool_functions: dict | None = None,
    **kwargs,
) -> dict:
    """
    Run a harness with full Phoenix tracing.

    Wraps the clean harness and creates spans from the returned iteration data.
    """
    harness_fn = HARNESSES[harness_name]
    harness_type = harness_name  # Name is now the type: implicit, explicit, adaptive

    # Run the harness (clean, no tracing inside)
    result = harness_fn(
        prompt=prompt,
        provider=provider,
        model=model,
        tools_list=tools_list,
        tool_functions=tool_functions,
        **kwargs,
    )

    # Now create spans from the result data
    with _chain_span(
        "agent_run",
        harness_type=harness_type,
        model_provider=provider,
        model_name=model or "default",
        **{"input.value": prompt},
    ) as chain:
        # Create iteration spans from the recorded data
        for iter_data in result["iterations"]:
            with _iteration_span(
                iter_data["n"],
                finish_triggered=iter_data.get("finish_triggered", False),
                false_finish=iter_data.get("false_finish", False),
                narrate_then_act=iter_data.get("narrate_then_act", False),
            ):
                # Create tool spans
                for tool in iter_data.get("tool_calls", []):
                    with _tool_span(tool["name"], tool["arguments"], tool["result"]):
                        pass  # Span created, nothing to do inside

        # Set final attributes on chain span
        chain.set_attribute("output.value", result["text"])
        chain.set_attribute("task_complete", result["task_complete"])
        chain.set_attribute("iterations", result["iteration_count"])
        chain.set_attribute("tool_calls_total", result["tool_calls_total"])
        chain.set_attribute("finished_reason", result["finished_reason"])

    return result


def run_eval_matrix(
    prompts: list[str],
    harnesses: list[str] = ["A", "B", "C"],
    providers: list[str] = ["anthropic"],
    tools_list: list | None = None,
    tool_functions: dict | None = None,
):
    """Run evaluation across harnesses and providers."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    results = []

    for prompt in prompts:
        for provider in providers:
            for harness in harnesses:
                console.print(f"Running Harness {harness} | {provider}...")

                result = run_with_tracing(
                    harness_name=harness,
                    prompt=prompt,
                    provider=provider,
                    tools_list=tools_list,
                    tool_functions=tool_functions,
                )
                result["harness"] = harness
                result["provider"] = provider
                results.append(result)

    # Print summary table
    table = Table(title="Results")
    table.add_column("Harness")
    table.add_column("Provider")
    table.add_column("Complete")
    table.add_column("Iterations")
    table.add_column("Tools")
    table.add_column("Tokens")
    table.add_column("Finished")

    for r in results:
        table.add_row(
            r["harness"],
            r["provider"],
            "Yes" if r["task_complete"] else "No",
            str(r["iteration_count"]),
            str(r["tool_calls_total"]),
            f"{r['input_tokens'] + r['output_tokens']:,}",
            r["finished_reason"],
        )

    console.print(table)
    return results


# Example usage
if __name__ == "__main__":
    init_tracing()

    # Import GAIA tools
    from tools_gaia import tools_list, tool_functions

    result = run_with_tracing(
        harness_name="C",
        prompt="What is 2 + 2? Use the calculator.",
        provider="anthropic",
        tools_list=tools_list,
        tool_functions=tool_functions,
    )

    print(f"Complete: {result['task_complete']}")
    print(f"Iterations: {result['iteration_count']}")
