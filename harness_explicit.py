"""
Explicit Finish Harness

Finish condition: Model must explicitly call finish() tool.
- finish() validates that model produced text output
- Text-only responses get nudged to call finish()
- Max 3 failed finish attempts before force-exit

Usage:
    python harness_explicit.py --provider anthropic --model claude-sonnet-4-20250514
    python harness_explicit.py --provider openai --model gpt-4o
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
from tools import tool_functions as default_tool_functions
from tools import tools_list as default_tools_list
from tools import reset_tasks as default_reset_tasks

console = Console()

# Default system prompt (Jose-style: encourages acknowledgement + planning)
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

When you have completed the user's request, call the finish() tool to end the session."""

# Default test prompt (multi-step task that can trigger false finishes)
DEFAULT_PROMPT = "Can you find the populations of Tokyo, Osaka, and Yokohama, sort them by size, multiply each by 314, and show me the results in a Markdown table?"


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
    tools_list: list | None = None,
    tool_functions: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the agent loop with explicit finish requirement.

    The model must call finish() to end the session. This ensures intentional
    completion rather than accidental exits from text-only responses.

    Returns:
        dict with iteration details for optional external tracing
    """
    # Use provided tools or defaults
    _tools_list = tools_list if tools_list is not None else default_tools_list
    _tool_functions = tool_functions if tool_functions is not None else default_tool_functions

    default_reset_tasks()

    model = model or DEFAULT_MODELS[provider]
    messages = [{"role": "user", "content": prompt}]

    # Add finish tool to the tool set
    harness_tools = _tools_list + [FINISH_TOOL]

    iterations = []
    all_text = ""
    finish_attempts = 0
    finished_reason = "max_iterations"

    if verbose:
        console.print(Rule(style="white"))
        console.print(Panel(prompt, title="[bold white]User Request[/bold white]", border_style="white"))
        console.print(Rule(style="white"))
        console.print(f"[dim]Provider: {provider} | Model: {model} | Max iterations: {max_iterations}[/dim]\n")

    for i in range(1, max_iterations + 1):
        iter_data = {
            "n": i,
            "tool_calls": [],
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_triggered": False,
            "finish_rejected": False,
            "nudge": None,
        }

        if verbose:
            console.print(Rule(Text(f" Iteration {i} ", style="bold cyan"), style="cyan"))

        response: ModelResponse = call_model(
            provider=provider,
            model=model,
            messages=messages,
            tools=harness_tools,
            system_prompt=system_prompt,
        )

        iter_data["input_tokens"] = response.tokens.input_tokens
        iter_data["output_tokens"] = response.tokens.output_tokens

        if response.text:
            if verbose:
                console.print(Text("🤖 Assistant:", style="bold green"))
                console.print(Markdown(response.text))
                console.print()
            iter_data["text"] = response.text
            all_text += response.text + "\n"

        # ─────────────────────────────────────────────────────────────
        # TEXT-ONLY RESPONSE: Nudge to call finish()
        # ─────────────────────────────────────────────────────────────
        if not response.tool_calls:
            nudge = "If you're done, call the finish() tool to end the session."
            if verbose:
                console.print(Text(f"💬 NUDGE: {nudge}", style="bold magenta"))
            iter_data["nudge"] = nudge
            messages.append(format_assistant_message(provider, response))
            messages.append({"role": "user", "content": nudge})
            iterations.append(iter_data)
            continue

        if verbose:
            console.print(Text(f"🛠️  Tool calls ({len(response.tool_calls)}):", style="bold yellow"))

        messages.append(format_assistant_message(provider, response))

        should_break = False
        for tool_call in response.tool_calls:
            name = tool_call.name
            args = tool_call.arguments

            if verbose:
                console.print(f"   [yellow]→ {name}[/yellow]")
                args_str = json.dumps(args, indent=2)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                console.print(f"     [dim]Input:[/dim] {args_str}")

            # ─────────────────────────────────────────────────────────
            # FINISH TOOL: Validate before accepting
            # ─────────────────────────────────────────────────────────
            if name == "finish":
                finish_attempts += 1
                summary = args.get("summary", "")

                can_finish, validation_msg = _validate_finish(all_text, summary)

                if can_finish:
                    if verbose:
                        console.print(f"     [bold green]✅ {validation_msg}[/bold green]")
                    iter_data["finish_triggered"] = True
                    finished_reason = "finish_validated"
                    messages.append(
                        format_tool_result_message(provider, tool_call.id, name, "Session ended successfully.")
                    )
                    iter_data["tool_calls"].append({
                        "name": name,
                        "arguments": args,
                        "result": "Session ended successfully.",
                    })
                    should_break = True
                    break
                else:
                    if verbose:
                        console.print(f"     [bold red]❌ Validation failed:[/bold red]")
                        console.print(f"     {validation_msg}")
                    iter_data["finish_rejected"] = True

                    if finish_attempts >= max_finish_attempts:
                        finished_reason = "max_finish_attempts"
                        if verbose:
                            console.print(Text(f"     ⚠️ Max finish attempts reached. Force-exiting.", style="bold red"))
                        should_break = True
                        break

                    messages.append(
                        format_tool_result_message(provider, tool_call.id, name, validation_msg)
                    )
                    iter_data["tool_calls"].append({
                        "name": name,
                        "arguments": args,
                        "result": validation_msg,
                    })
                continue

            # Regular tool execution
            if name in _tool_functions:
                try:
                    result = _tool_functions[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {name}"

            iter_data["tool_calls"].append({
                "name": name,
                "arguments": args,
                "result": str(result)[:500],
            })

            if verbose:
                result_str = str(result)
                if len(result_str) > 300:
                    result_str = result_str[:300] + "..."
                console.print(f"     [dim]Result:[/dim] {result_str}")

            messages.append(
                format_tool_result_message(provider, tool_call.id, name, str(result))
            )

        iterations.append(iter_data)

        if should_break:
            if verbose:
                console.print(Rule(style="white"))
                if finished_reason == "finish_validated":
                    console.print(Text("✅ EXIT: finish() validated", style="bold green"))
            break

        if verbose:
            console.print()

    # Final summary
    if verbose:
        console.print(Rule(style="white"))
        if finished_reason == "max_iterations":
            console.print(Text(f"⚠️  EXIT: Max iterations ({max_iterations}) reached", style="bold red"))

        total_input = sum(it["input_tokens"] for it in iterations)
        total_output = sum(it["output_tokens"] for it in iterations)
        false_finishes = max(0, finish_attempts - 1) if finished_reason == "finish_validated" else finish_attempts

        console.print(Panel(
            f"[bold]Iterations:[/bold] {len(iterations)}\n"
            f"[bold]Tool calls:[/bold] {sum(len(it['tool_calls']) for it in iterations)}\n"
            f"[bold]Tokens:[/bold] {total_input + total_output:,} ({total_input:,} in / {total_output:,} out)\n"
            f"[bold]Finish attempts:[/bold] {finish_attempts}\n"
            f"[bold]Finished:[/bold] {finished_reason}",
            title="[bold]Summary[/bold]",
            border_style="cyan",
        ))

    false_finishes = max(0, finish_attempts - 1) if finished_reason == "finish_validated" else finish_attempts

    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "iteration_count": len(iterations),
        "tool_calls_total": sum(len(it["tool_calls"]) for it in iterations),
        "input_tokens": sum(it["input_tokens"] for it in iterations),
        "output_tokens": sum(it["output_tokens"] for it in iterations),
        "total_tokens": sum(it["input_tokens"] + it["output_tokens"] for it in iterations),
        "false_finishes": false_finishes,
        "narrate_then_act": 0,
        "finish_attempts": finish_attempts,
        "task_complete": finished_reason == "finish_validated",
        "finished_reason": finished_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Explicit Finish Harness"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "openai", "openrouter"],
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
