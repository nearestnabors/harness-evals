"""
Harness A: Implicit finish (Claude Code style)

Finish condition: The model stops calling tools (no tool_calls in response).
This is the simplest approach - the model decides when it's done.

Usage:
    python harness_a.py --provider anthropic --model claude-sonnet-4-20250514
    python harness_a.py --provider openai --model gpt-4o
"""

import argparse
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Initialize tracing BEFORE importing models (which creates LLM clients)
from instrumentation import (
    init_tracing,
    shutdown_tracing,
    chain_span,
    iteration_span,
    tool_span,
    set_span_output,
    ATTR_FINISH_TRIGGERED,
    ATTR_TASK_COMPLETE,
    ATTR_OUTPUT_VALUE,
)

from models import (
    ModelResponse,
    Provider,
    call_model,
    format_assistant_message,
    format_tool_result_message,
    DEFAULT_MODELS,
)
from tools import tool_functions, tools_list, reset_tasks

console = Console()

# Default system prompt
SYSTEM_PROMPT = """You are an agent that helps analyze AI system traces and spans.
You have access to tools to search and analyze trace data.
Be conversational and explain what you're doing as you work.
When you have completed the user's request, provide your final answer."""

# Default test prompt
DEFAULT_PROMPT = "Can you find all the span ids that contain the word 'Aragorn', sort them by latency, multiply the latency by 314 and show the result in Markdown."


def run(
    prompt: str,
    provider: Provider = "anthropic",
    model: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_iterations: int = 20,
) -> dict:
    """
    Run the agent loop with implicit finish condition.

    Loop:
        1. Send messages to the model
        2. If response has tool calls → execute them, append results, continue
        3. If response has text but NO tool calls → print text and exit
        4. Safety: max 20 iterations

    Returns:
        dict with keys:
            - text: final response text
            - iterations: number of loop iterations
            - tool_calls_total: total number of tool calls made
            - finished_reason: "no_tool_calls" or "max_iterations"
    """
    reset_tasks()

    model = model or DEFAULT_MODELS[provider]
    messages = [{"role": "user", "content": prompt}]

    iterations = 0
    tool_calls_total = 0
    total_input_tokens = 0
    total_output_tokens = 0
    all_text = ""
    finished_reason = "max_iterations"

    # Log the start
    console.print(Rule(style="white"))
    console.print(Panel(prompt, title="[bold white]User Request[/bold white]", border_style="white"))
    console.print(Rule(style="white"))
    console.print(f"[dim]Provider: {provider} | Model: {model} | Max iterations: {max_iterations}[/dim]\n")

    # Wrap entire agent run in a CHAIN span
    with chain_span(
        name="run_agent",
        harness_type="implicit",
        model_provider=provider,
        model_name=model,
        input_value=prompt,
    ) as agent_span:

        while iterations < max_iterations:
            iterations += 1

            # Wrap each iteration in its own span
            with iteration_span(
                iteration_number=iterations,
                harness_type="implicit",
                model_provider=provider,
                model_name=model,
            ) as iter_span:

                # ─────────────────────────────────────────────────────────────
                # Iteration header
                # ─────────────────────────────────────────────────────────────
                console.print(Rule(Text(f" Iteration {iterations} ", style="bold cyan"), style="cyan"))

                # ─────────────────────────────────────────────────────────────
                # Step 1: Send messages to the model
                # ─────────────────────────────────────────────────────────────
                response: ModelResponse = call_model(
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tools_list,
                    system_prompt=system_prompt,
                )

                # Accumulate token usage
                total_input_tokens += response.tokens.input_tokens
                total_output_tokens += response.tokens.output_tokens

                # ─────────────────────────────────────────────────────────────
                # Log what the model said (narration)
                # ─────────────────────────────────────────────────────────────
                if response.text:
                    console.print(Text("🤖 Assistant (narration):", style="bold green"))
                    console.print(Markdown(response.text))
                    console.print()
                    all_text += response.text + "\n"

                # ─────────────────────────────────────────────────────────────
                # Step 3: If NO tool calls → exit
                # ─────────────────────────────────────────────────────────────
                if not response.tool_calls:
                    console.print(Text("✅ EXIT: No tool calls in response", style="bold green"))
                    finished_reason = "no_tool_calls"

                    # Mark this iteration as the finish trigger
                    iter_span.set_attribute(ATTR_FINISH_TRIGGERED, True)
                    set_span_output(iter_span, response.text)
                    break

                # ─────────────────────────────────────────────────────────────
                # Log tool calls (action)
                # ─────────────────────────────────────────────────────────────
                console.print(Text(f"🛠️  Tool calls ({len(response.tool_calls)}):", style="bold yellow"))

                # Add assistant message to history (before executing tools)
                messages.append(format_assistant_message(provider, response))

                # ─────────────────────────────────────────────────────────────
                # Step 2: Execute tool calls, append results, continue
                # ─────────────────────────────────────────────────────────────
                for tool_call in response.tool_calls:
                    tool_calls_total += 1
                    name = tool_call.name
                    args = tool_call.arguments

                    # Log the tool call
                    console.print(f"   [yellow]→ {name}[/yellow]")
                    args_str = json.dumps(args, indent=2)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    console.print(f"     [dim]Input:[/dim] {args_str}")

                    # Execute the tool with tracing
                    with tool_span(name, args) as t_span:
                        if name in tool_functions:
                            try:
                                result = tool_functions[name](**args)
                            except Exception as e:
                                result = f"Error: {e}"
                        else:
                            result = f"Unknown tool: {name}"

                        # Set tool output on span
                        set_span_output(t_span, str(result))

                    # Log the result
                    result_str = str(result)
                    if len(result_str) > 300:
                        result_preview = result_str[:300] + "..."
                    else:
                        result_preview = result_str
                    console.print(f"     [dim]Result:[/dim] {result_preview}")

                    # Append tool result to messages
                    messages.append(
                        format_tool_result_message(provider, tool_call.id, name, str(result))
                    )

                console.print()  # Blank line before next iteration

        # ─────────────────────────────────────────────────────────────
        # Final summary
        # ─────────────────────────────────────────────────────────────
        console.print(Rule(style="white"))

        if finished_reason == "max_iterations":
            console.print(Text(f"⚠️  EXIT: Max iterations ({max_iterations}) reached", style="bold red"))

        console.print(Panel(
            f"[bold]Iterations:[/bold] {iterations}\n"
            f"[bold]Tool calls:[/bold] {tool_calls_total}\n"
            f"[bold]Tokens:[/bold] {total_input_tokens + total_output_tokens:,} ({total_input_tokens:,} in / {total_output_tokens:,} out)\n"
            f"[bold]Finished:[/bold] {finished_reason}",
            title="[bold]Summary[/bold]",
            border_style="cyan",
        ))

        # Set final attributes on the agent span
        set_span_output(agent_span, all_text.strip())
        agent_span.set_attribute(ATTR_TASK_COMPLETE, finished_reason == "no_tool_calls")

    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "tool_calls_total": tool_calls_total,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "false_finishes": 0,  # Harness A doesn't track false finishes
        "narrate_then_act": 0,  # Harness A doesn't track narrate-then-act
        "task_complete": finished_reason == "no_tool_calls",
        "finished_reason": finished_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Harness A: Implicit finish (Claude Code style)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Model provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name (default: provider's default)",
    )
    parser.add_argument(
        "--prompt", "-P",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--max-iterations", "-i",
        type=int,
        default=20,
        help="Max iterations before forced exit (default: 20)",
    )

    args = parser.parse_args()

    # Initialize tracing before running
    init_tracing()

    result = run(
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
    )

    # Ensure spans are flushed before exit
    shutdown_tracing()

    return result


if __name__ == "__main__":
    main()
