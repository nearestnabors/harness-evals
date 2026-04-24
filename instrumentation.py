"""
Phoenix/OpenInference instrumentation for the finish-condition eval harnesses.

This module:
1. Launches a local Phoenix server for trace visualization
2. Registers auto-instrumentors for Anthropic and OpenAI
3. Provides helpers for manual CHAIN and TOOL spans with custom attributes

IMPORTANT: Call init_tracing() BEFORE importing models.py or creating any LLM clients.
"""

import json
from contextlib import contextmanager
from typing import Any, Literal

from opentelemetry.trace import get_tracer, Status, StatusCode

# Custom attribute keys for eval analysis
ATTR_HARNESS_TYPE = "harness_type"
ATTR_MODEL_PROVIDER = "model_provider"
ATTR_MODEL_NAME = "model_name"
ATTR_ITERATION_NUMBER = "iteration_number"
ATTR_FINISH_TRIGGERED = "finish_triggered"
ATTR_FALSE_FINISH = "false_finish"
ATTR_NARRATE_THEN_ACT = "narrate_then_act"
ATTR_TASK_COMPLETE = "task_complete"
ATTR_TODOS_ABANDONED = "todos_abandoned"

# OpenInference semantic convention attributes
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_INPUT_VALUE = "input.value"
ATTR_OUTPUT_VALUE = "output.value"

# Module-level state
_tracer = None
_initialized = False


HarnessType = Literal["implicit", "explicit", "adaptive"]


def init_tracing(project_name: str = "harness-evals") -> bool:
    """
    Initialize local Phoenix tracing.

    Launches a Phoenix server at http://localhost:6006 and configures
    OpenTelemetry to send traces there.

    Must be called BEFORE importing models.py or creating any LLM clients.

    Args:
        project_name: Name of the project in Phoenix UI

    Returns:
        True if initialization succeeded, False otherwise
    """
    global _tracer, _initialized

    if _initialized:
        return True

    try:
        from phoenix.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor

        # Register the tracer provider to send to Phoenix
        # Assumes Phoenix is already running: python -m phoenix.server.main serve
        register(
            project_name=project_name,
            endpoint="http://localhost:6006/v1/traces",
        )

        # Instrument Anthropic and OpenAI clients
        # These will auto-create LLM spans for every API call
        AnthropicInstrumentor().instrument()
        OpenAIInstrumentor().instrument()

        # Get a tracer for our manual spans
        _tracer = get_tracer("harness-evals", "1.0.0")
        _initialized = True

        print(f"[instrumentation] Tracing to Phoenix at http://localhost:6006")
        print(f"[instrumentation] Project: {project_name}")
        return True

    except Exception as e:
        print(f"[instrumentation] ERROR initializing tracing: {e}")
        return False


def shutdown_tracing():
    """
    Flush and shutdown the tracer provider.

    Call this before the process exits to ensure all spans are exported.
    """
    try:
        from opentelemetry.trace import get_tracer_provider

        provider = get_tracer_provider()
        if hasattr(provider, 'force_flush'):
            provider.force_flush()
        if hasattr(provider, 'shutdown'):
            provider.shutdown()

        print("[instrumentation] Tracing shutdown complete")
    except Exception as e:
        print(f"[instrumentation] WARNING: Error during shutdown: {e}")


@contextmanager
def chain_span(
    name: str,
    harness_type: HarnessType,
    model_provider: str,
    model_name: str,
    input_value: str,
):
    """
    Context manager for a CHAIN span (wraps the full agent run).

    Usage:
        with chain_span("run_agent", "implicit", "anthropic", "claude-sonnet-4-20250514", prompt) as span:
            # ... agent loop ...
            span.set_attribute("output.value", final_text)
            span.set_attribute("task_complete", True)

    Args:
        name: Span name (e.g., "run_agent")
        harness_type: "implicit" | "explicit" | "adaptive"
        model_provider: "anthropic" | "openai"
        model_name: Specific model name
        input_value: User prompt / input

    Yields:
        The span object for setting additional attributes
    """
    if not _tracer:
        # Tracing not initialized - yield a no-op context
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def set_status(self, status): pass
            def record_exception(self, exc): pass
        yield NoOpSpan()
        return

    with _tracer.start_as_current_span(name) as span:
        # Set OpenInference span kind
        span.set_attribute(ATTR_SPAN_KIND, "CHAIN")
        span.set_attribute(ATTR_INPUT_VALUE, input_value)

        # Set custom eval attributes
        span.set_attribute(ATTR_HARNESS_TYPE, harness_type)
        span.set_attribute(ATTR_MODEL_PROVIDER, model_provider)
        span.set_attribute(ATTR_MODEL_NAME, model_name)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def iteration_span(
    iteration_number: int,
    harness_type: HarnessType,
    model_provider: str,
    model_name: str,
):
    """
    Context manager for an iteration span (one loop iteration).

    Creates a CHAIN span for a single iteration of the agent loop.

    Args:
        iteration_number: Which iteration (1-indexed)
        harness_type: "implicit" | "explicit" | "adaptive"
        model_provider: "anthropic" | "openai"
        model_name: Specific model name

    Yields:
        The span object for setting additional attributes
    """
    if not _tracer:
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def set_status(self, status): pass
            def record_exception(self, exc): pass
        yield NoOpSpan()
        return

    with _tracer.start_as_current_span(f"iteration_{iteration_number}") as span:
        span.set_attribute(ATTR_SPAN_KIND, "CHAIN")
        span.set_attribute(ATTR_HARNESS_TYPE, harness_type)
        span.set_attribute(ATTR_MODEL_PROVIDER, model_provider)
        span.set_attribute(ATTR_MODEL_NAME, model_name)
        span.set_attribute(ATTR_ITERATION_NUMBER, iteration_number)

        # These will be set by the harness as the iteration progresses
        span.set_attribute(ATTR_FINISH_TRIGGERED, False)
        span.set_attribute(ATTR_FALSE_FINISH, False)
        span.set_attribute(ATTR_NARRATE_THEN_ACT, False)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def tool_span(
    tool_name: str,
    tool_input: dict[str, Any],
):
    """
    Context manager for a TOOL span (wraps a single tool invocation).

    Usage:
        with tool_span("calculator", {"expression": "2+2"}) as span:
            result = calculator("2+2")
            span.set_attribute("output.value", result)

    Args:
        tool_name: Name of the tool being called
        tool_input: Arguments passed to the tool

    Yields:
        The span object for setting the output
    """
    if not _tracer:
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def set_status(self, status): pass
            def record_exception(self, exc): pass
        yield NoOpSpan()
        return

    with _tracer.start_as_current_span(tool_name) as span:
        span.set_attribute(ATTR_SPAN_KIND, "TOOL")
        span.set_attribute(ATTR_INPUT_VALUE, json.dumps(tool_input))

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def set_span_output(span, output_value: Any):
    """Helper to set the output value on a span."""
    if span and hasattr(span, 'set_attribute'):
        if isinstance(output_value, str):
            span.set_attribute(ATTR_OUTPUT_VALUE, output_value)
        else:
            span.set_attribute(ATTR_OUTPUT_VALUE, json.dumps(output_value))
