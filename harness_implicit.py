"""
Implicit Finish Harness

Finish condition: The model stops calling tools (no tool_calls in response).
This is the simplest approach - the model decides when it's done.

Usage:
    python harness_implicit.py --provider anthropic --model claude-sonnet-4-20250514
    python harness_implicit.py --provider openai --model gpt-4o
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
Respond politely and conversationally, starting with a brief acknowledgement."""

# Default test prompt (multi-step task that can trigger false finishes)
DEFAULT_PROMPT = "Can you find the populations of Tokyo, Osaka, and Yokohama, sort them by size, multiply each by 314, and show me the results in a Markdown table?"


def run(
    prompt: str,
    provider: Provider = "anthropic",
    model: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_iterations: int = 20,
    tools_list: list | None = None,
    tool_functions: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the agent loop with implicit finish condition.

    The simplest finish condition: if the model responds without tool calls,
    we assume it's done. This works well with Claude models which reliably
    combine narration + tool calls in a single response.

    Returns:
        dict with iteration details for optional external tracing
    """
    # Use provided tools or defaults
    _tools_list = tools_list if tools_list is not None else default_tools_list
    _tool_functions = tool_functions if tool_functions is not None else default_tool_functions

    default_reset_tasks()

    model = model or DEFAULT_MODELS[provider]
    messages = [{"role": "user", "content": prompt}]

    iterations = []
    all_text = ""
    finished_reason = "max_iterations"

    # Log the start
    if verbose:
        console.print(Rule(style="white"))
        console.print(Panel(prompt, title="[bold white]User Request[/bold white]", border_style="white"))
        console.print(Rule(style="white"))
        console.print(f"[dim]Provider: {provider} | Model: {model} | Max iterations: {max_iterations}[/dim]\n")

    for i in range(1, max_iterations + 1):
        # Track this iteration
        iter_data = {
            "n": i,
            "tool_calls": [],
            "text": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_triggered": False,
        }

        if verbose:
            console.print(Rule(Text(f" Iteration {i} ", style="bold cyan"), style="cyan"))

        # Call the model
        response: ModelResponse = call_model(
            provider=provider,
            model=model,
            messages=messages,
            tools=_tools_list,
            system_prompt=system_prompt,
        )

        iter_data["input_tokens"] = response.tokens.input_tokens
        iter_data["output_tokens"] = response.tokens.output_tokens

        # Log and capture text
        if response.text:
            if verbose:
                console.print(Text("🤖 Assistant:", style="bold green"))
                console.print(Markdown(response.text))
                console.print()
            iter_data["text"] = response.text
            all_text += response.text + "\n"

        # ─────────────────────────────────────────────────────────────
        # THE FINISH CONDITION: No tool calls = we're done
        # ─────────────────────────────────────────────────────────────
        if not response.tool_calls:
            if verbose:
                console.print(Text("✅ EXIT: No tool calls in response", style="bold green"))
            iter_data["finish_triggered"] = True
            finished_reason = "no_tool_calls"
            iterations.append(iter_data)
            break

        # Execute tool calls
        if verbose:
            console.print(Text(f"🛠️  Tool calls ({len(response.tool_calls)}):", style="bold yellow"))

        messages.append(format_assistant_message(provider, response))

        for tool_call in response.tool_calls:
            name = tool_call.name
            args = tool_call.arguments

            if verbose:
                console.print(f"   [yellow]→ {name}[/yellow]")
                args_str = json.dumps(args, indent=2)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                console.print(f"     [dim]Input:[/dim] {args_str}")

            # Execute the tool
            if name in _tool_functions:
                try:
                    result = _tool_functions[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {name}"

            # Track it
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
        if verbose:
            console.print()

    # Final summary
    if verbose:
        console.print(Rule(style="white"))
        if finished_reason == "max_iterations":
            console.print(Text(f"⚠️  EXIT: Max iterations ({max_iterations}) reached", style="bold red"))

        total_input = sum(it["input_tokens"] for it in iterations)
        total_output = sum(it["output_tokens"] for it in iterations)
        console.print(Panel(
            f"[bold]Iterations:[/bold] {len(iterations)}\n"
            f"[bold]Tool calls:[/bold] {sum(len(it['tool_calls']) for it in iterations)}\n"
            f"[bold]Tokens:[/bold] {total_input + total_output:,} ({total_input:,} in / {total_output:,} out)\n"
            f"[bold]Finished:[/bold] {finished_reason}",
            title="[bold]Summary[/bold]",
            border_style="cyan",
        ))

    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "iteration_count": len(iterations),
        "tool_calls_total": sum(len(it["tool_calls"]) for it in iterations),
        "input_tokens": sum(it["input_tokens"] for it in iterations),
        "output_tokens": sum(it["output_tokens"] for it in iterations),
        "total_tokens": sum(it["input_tokens"] + it["output_tokens"] for it in iterations),
        "false_finishes": 0,  # Harness A doesn't detect false finishes
        "narrate_then_act": 0,
        "task_complete": finished_reason == "no_tool_calls",
        "finished_reason": finished_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Implicit Finish Harness"
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

    result = run(
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
    )

    return result


if __name__ == "__main__":
    main()
