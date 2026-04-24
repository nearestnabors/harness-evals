"""
Harness C: Adaptive finish condition (hybrid)

Uses implicit finish as primary signal (like Harness A) but with a
lightweight false-finish detector:

1. If model responds with text + no tool calls, DON'T immediately exit
2. Check text for narrate-then-act signals (heuristic)
3. If detected: inject "Continue with your plan." and keep looping
4. If not detected: genuine completion, exit
5. Safety: max 3 consecutive false-finishes without tool calls → force-exit

Usage:
    python harness_c.py --provider anthropic --model claude-sonnet-4-20250514
    python harness_c.py --provider openai --model gpt-4o
"""

import argparse
import json
import re

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
    ATTR_FALSE_FINISH,
    ATTR_NARRATE_THEN_ACT,
    ATTR_TASK_COMPLETE,
    ATTR_TODOS_ABANDONED,
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
from tools import tool_functions, tools_list, reset_tasks, TASKS

console = Console()

# Default system prompt
SYSTEM_PROMPT = """You are an agent that helps analyze AI system traces and spans.
You have access to tools to search and analyze trace data.
Be conversational and explain what you're doing as you work.
When you have completed the user's request, provide your final answer."""

# Default test prompt
DEFAULT_PROMPT = "Can you find all the span ids that contain the word 'Aragorn', sort them by latency, multiply the latency by 314 and show the result in Markdown."


# ─────────────────────────────────────────────────────────────────────────────
# False-finish detection heuristics
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that suggest the model is about to do something (narrate-then-act)
# These indicate the model said what it's going to do but didn't actually do it
NARRATE_THEN_ACT_PATTERNS = [
    # "I'll search for...", "I'll look at..."
    r"\bI'?ll\s+\w+",
    # "I'm going to search...", "I'm going to look..."
    r"\bI'?m\s+going\s+to\s+\w+",
    # "Let me search...", "Let me check..."
    r"\bLet\s+me\s+\w+",
    # "Now I will...", "Now I'll..."
    r"\bNow\s+I\s+(will|'ll)\s+\w+",
    # "I need to search...", "I need to check..."
    r"\bI\s+need\s+to\s+\w+",
    # "Next, I will...", "Next I'll..."
    r"\bNext,?\s+I\s+(will|'ll)?\s*\w+",
    # "First, I will...", "First, let me..."
    r"\bFirst,?\s+(I\s+(will|'ll)|let\s+me)\s+\w+",
    # "I should search...", "I should check..."
    r"\bI\s+should\s+\w+",
    # "I can search...", "I can use..."
    r"\bI\s+can\s+\w+",
    # "I will search...", "I will look..."
    r"\bI\s+will\s+\w+",
    # "Going to search...", "Going to check..."
    r"\bGoing\s+to\s+\w+",
    # "To do this, I'll..."
    r"\bTo\s+do\s+this,?\s+I",
    # "My plan is to..."
    r"\bMy\s+plan\s+is\s+to\s+\w+",
]

# Compile patterns for efficiency
NARRATE_PATTERNS_COMPILED = [
    re.compile(pattern, re.IGNORECASE) for pattern in NARRATE_THEN_ACT_PATTERNS
]


def detect_narrate_then_act(text: str) -> tuple[bool, list[str]]:
    """
    Detect if the text contains narrate-then-act signals.

    Returns:
        (detected, matched_phrases) - whether detected and what phrases matched
    """
    if not text:
        return False, []

    matched = []
    for pattern in NARRATE_PATTERNS_COMPILED:
        matches = pattern.findall(text)
        if matches:
            # Get the actual matched text from the original
            for match in pattern.finditer(text):
                matched.append(match.group(0))

    return len(matched) > 0, matched[:3]  # Return up to 3 matches


def _count_incomplete_todos() -> int:
    """Count todos that are not marked as completed."""
    return sum(1 for task in TASKS if task.get("status") != "completed")


def run(
    prompt: str,
    provider: Provider = "anthropic",
    model: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_iterations: int = 20,
    max_consecutive_false_finishes: int = 3,
) -> dict:
    """
    Run the agent loop with adaptive finish condition.

    Loop:
        1. Send messages to the model
        2. If tool calls → execute them, reset false-finish counter, continue
        3. If text-only:
           a. Check for narrate-then-act signals
           b. If detected → inject "Continue" nudge, increment counter
           c. If not detected → genuine finish, exit
        4. Safety: max consecutive false-finishes → force-exit

    Returns:
        dict with results
    """
    reset_tasks()

    model = model or DEFAULT_MODELS[provider]
    messages = [{"role": "user", "content": prompt}]

    iterations = 0
    tool_calls_total = 0
    false_finish_count = 0  # Consecutive false finishes
    total_false_finishes = 0  # Total false finishes detected
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
        harness_type="adaptive",
        model_provider=provider,
        model_name=model,
        input_value=prompt,
    ) as agent_span:

        while iterations < max_iterations:
            iterations += 1

            # Wrap each iteration in its own span
            with iteration_span(
                iteration_number=iterations,
                harness_type="adaptive",
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

                # ─────────────────────────────────────────────────────────────
                # Log what the model said (narration)
                # ─────────────────────────────────────────────────────────────
                if response.text:
                    console.print(Text("🤖 Assistant (narration):", style="bold green"))
                    console.print(Markdown(response.text))
                    console.print()
                    all_text += response.text + "\n"

                # ─────────────────────────────────────────────────────────────
                # Handle: No tool calls (potential finish)
                # ─────────────────────────────────────────────────────────────
                if not response.tool_calls:
                    # Check for narrate-then-act signals
                    is_false_finish, matched_phrases = detect_narrate_then_act(response.text)

                    if is_false_finish:
                        false_finish_count += 1
                        total_false_finishes += 1

                        console.print(Text("🔍 FALSE-FINISH DETECTED:", style="bold magenta"))
                        console.print(f"   [dim]Matched phrases:[/dim] {matched_phrases}")
                        console.print(f"   [dim]Consecutive:[/dim] {false_finish_count}/{max_consecutive_false_finishes}")

                        # Mark iteration span with false finish info
                        iter_span.set_attribute(ATTR_FALSE_FINISH, True)
                        iter_span.set_attribute(ATTR_NARRATE_THEN_ACT, True)

                        # Safety: too many consecutive false finishes
                        if false_finish_count >= max_consecutive_false_finishes:
                            finished_reason = "stuck_narration_loop"
                            console.print(
                                Text(f"   ⚠️ Max consecutive false-finishes reached. Model may be stuck.",
                                     style="bold red")
                            )
                            break

                        # Inject nudge to continue
                        nudge = "Continue with your plan."
                        console.print(Text(f"   💬 NUDGE: {nudge}", style="bold magenta"))

                        messages.append(format_assistant_message(provider, response))
                        messages.append({"role": "user", "content": nudge})
                        continue

                    else:
                        # No narrate-then-act signal → genuine completion
                        console.print(Text("✅ EXIT: Genuine completion (no action signals detected)", style="bold green"))
                        finished_reason = "genuine_completion"

                        # Mark this iteration as the finish trigger
                        iter_span.set_attribute(ATTR_FINISH_TRIGGERED, True)
                        set_span_output(iter_span, response.text)
                        break

                # ─────────────────────────────────────────────────────────────
                # Has tool calls → reset false-finish counter
                # ─────────────────────────────────────────────────────────────
                if false_finish_count > 0:
                    console.print(Text(f"   [dim]Reset false-finish counter (was {false_finish_count})[/dim]"))
                false_finish_count = 0

                # ─────────────────────────────────────────────────────────────
                # Log tool calls
                # ─────────────────────────────────────────────────────────────
                console.print(Text(f"🛠️  Tool calls ({len(response.tool_calls)}):", style="bold yellow"))

                # Add assistant message to history
                messages.append(format_assistant_message(provider, response))

                # ─────────────────────────────────────────────────────────────
                # Execute tool calls with tracing
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

        # Count abandoned todos
        todos_abandoned = _count_incomplete_todos()

        console.print(Panel(
            f"[bold]Iterations:[/bold] {iterations}\n"
            f"[bold]Tool calls:[/bold] {tool_calls_total}\n"
            f"[bold]False finishes detected:[/bold] {total_false_finishes}\n"
            f"[bold]Todos abandoned:[/bold] {todos_abandoned}\n"
            f"[bold]Finished:[/bold] {finished_reason}",
            title="[bold]Summary[/bold]",
            border_style="cyan",
        ))

        # Set final attributes on the agent span
        set_span_output(agent_span, all_text.strip())
        agent_span.set_attribute(ATTR_TASK_COMPLETE, finished_reason == "genuine_completion")
        agent_span.set_attribute(ATTR_TODOS_ABANDONED, todos_abandoned)

    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "tool_calls_total": tool_calls_total,
        "false_finishes_detected": total_false_finishes,
        "todos_abandoned": todos_abandoned,
        "finished_reason": finished_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Harness C: Adaptive finish condition (hybrid)"
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
    parser.add_argument(
        "--max-false-finishes",
        type=int,
        default=3,
        help="Max consecutive false-finishes before force-exit (default: 3)",
    )

    args = parser.parse_args()

    # Initialize tracing before running
    init_tracing()

    result = run(
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        max_consecutive_false_finishes=args.max_false_finishes,
    )

    # Ensure spans are flushed before exit
    shutdown_tracing()

    return result


if __name__ == "__main__":
    main()
