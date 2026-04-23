"""
Harness B: Explicit finish (simplified)

Finish condition: Model must explicitly call finish() tool.
- finish() validates that model produced text output
- Text-only responses get nudged to call finish()
- Max 3 failed finish attempts before force-exit
- No todo tracking required

Usage:
    python harness_b.py --provider anthropic --model claude-sonnet-4-20250514
    python harness_b.py --provider openai --model gpt-4o
"""

import argparse
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

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

# Default system prompt - simple, no todo requirements
SYSTEM_PROMPT = """You are an agent that helps analyze AI system traces and spans.
You have access to tools to search and analyze trace data.
Be conversational and explain what you're doing as you work.

When you have completed the user's request, call the finish() tool to end the session."""

# Default test prompt
DEFAULT_PROMPT = "Can you find all the span ids that contain the word 'Aragorn', sort them by latency, multiply the latency by 314 and show the result in Markdown."


# ─────────────────────────────────────────────────────────────────────────────
# Finish tool definition
# ─────────────────────────────────────────────────────────────────────────────

FINISH_TOOL = {
    "name": "finish",
    "description": "Call this tool when you have completed the user's request and are ready to end the session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Brief summary of what was accomplished.",
            }
        },
        "required": ["summary"],
    },
}


def _validate_finish(accumulated_text: str, summary: str) -> tuple[bool, str]:
    """
    Validate if the agent can finish.

    Only checks: did the model produce any output for the user?

    Returns:
        (can_finish, message)
    """
    if not accumulated_text.strip() and not summary.strip():
        return False, "You haven't provided any response to the user yet. Please share your findings before finishing."

    return True, "Finish validated successfully."


def run(
    prompt: str,
    provider: Provider = "anthropic",
    model: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_iterations: int = 20,
    max_finish_attempts: int = 3,
) -> dict:
    """
    Run the agent loop with explicit finish requirement.

    Loop:
        1. Send messages to the model
        2. If finish() called → validate output exists, allow/reject
        3. If other tool calls → execute them, continue
        4. If text-only → inject nudge to call finish()
        5. Safety: max iterations and max finish attempts

    Returns:
        dict with results
    """
    reset_tasks()

    model = model or DEFAULT_MODELS[provider]
    messages = [{"role": "user", "content": prompt}]

    # Add finish tool to the tool set
    harness_tools = tools_list + [FINISH_TOOL]

    iterations = 0
    tool_calls_total = 0
    finish_attempts = 0
    all_text = ""
    finished_reason = "max_iterations"

    # Log the start
    console.print(Rule(style="white"))
    console.print(Panel(prompt, title="[bold white]User Request[/bold white]", border_style="white"))
    console.print(Rule(style="white"))
    console.print(f"[dim]Provider: {provider} | Model: {model} | Max iterations: {max_iterations}[/dim]\n")

    while iterations < max_iterations:
        iterations += 1

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
            tools=harness_tools,
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
        # Handle: No tool calls (text-only response)
        # ─────────────────────────────────────────────────────────────
        if not response.tool_calls:
            nudge = "If you're done, call the finish() tool to end the session."
            console.print(Text(f"💬 NUDGE: {nudge}", style="bold magenta"))

            # Inject nudge as user message
            messages.append(format_assistant_message(provider, response))
            messages.append({"role": "user", "content": nudge})
            continue

        # ─────────────────────────────────────────────────────────────
        # Log tool calls
        # ─────────────────────────────────────────────────────────────
        console.print(Text(f"🛠️  Tool calls ({len(response.tool_calls)}):", style="bold yellow"))

        # Add assistant message to history
        messages.append(format_assistant_message(provider, response))

        # ─────────────────────────────────────────────────────────────
        # Execute tool calls
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

            # ─────────────────────────────────────────────────────────
            # Special handling for finish() tool
            # ─────────────────────────────────────────────────────────
            if name == "finish":
                finish_attempts += 1
                summary = args.get("summary", "")

                can_finish, validation_msg = _validate_finish(all_text, summary)

                if can_finish:
                    console.print(f"     [bold green]✅ {validation_msg}[/bold green]")
                    finished_reason = "finish_validated"

                    # Append success result
                    messages.append(
                        format_tool_result_message(
                            provider, tool_call.id, name,
                            "Session ended successfully."
                        )
                    )

                    # Exit the loop
                    console.print(Rule(style="white"))
                    console.print(Text("✅ EXIT: finish() validated", style="bold green"))
                    break
                else:
                    console.print(f"     [bold red]❌ Validation failed:[/bold red]")
                    console.print(f"     {validation_msg}")

                    if finish_attempts >= max_finish_attempts:
                        finished_reason = "max_finish_attempts"
                        console.print(
                            Text(f"     ⚠️ Max finish attempts ({max_finish_attempts}) reached. Force-exiting.",
                                 style="bold red")
                        )
                        break

                    # Send validation error back to model
                    messages.append(
                        format_tool_result_message(
                            provider, tool_call.id, name, validation_msg
                        )
                    )
                continue

            # ─────────────────────────────────────────────────────────
            # Execute regular tools
            # ─────────────────────────────────────────────────────────
            if name in tool_functions:
                try:
                    result = tool_functions[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {name}"

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

        # Check if we broke out of the tool loop due to finish
        if finished_reason in ("finish_validated", "max_finish_attempts"):
            break

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
        f"[bold]Finish attempts:[/bold] {finish_attempts}\n"
        f"[bold]Finished:[/bold] {finished_reason}",
        title="[bold]Summary[/bold]",
        border_style="cyan",
    ))

    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "tool_calls_total": tool_calls_total,
        "finish_attempts": finish_attempts,
        "finished_reason": finished_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Harness B: Explicit finish (simplified)"
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
        "--max-finish-attempts",
        type=int,
        default=3,
        help="Max failed finish() attempts before force-exit (default: 3)",
    )

    args = parser.parse_args()

    result = run(
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        max_finish_attempts=args.max_finish_attempts,
    )

    return result


if __name__ == "__main__":
    main()
