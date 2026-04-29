"""
Adaptive Finish Harness

Finish condition: Smart detection of genuine completion vs. false finishes.
- Detects "narrate-then-act" patterns ("I'll search for..." without calling tools)
- Recognizes completion signals ("Final Answer:", checkmarks) to avoid false positives
- Uses escalating nudges when model appears stuck
- Falls back to force-exit after 3 consecutive false finishes

Usage:
    python harness_adaptive.py --provider anthropic --model claude-sonnet-4-20250514
    python harness_adaptive.py --provider openai --model gpt-4o
"""

import argparse
import json
import re

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


# ─────────────────────────────────────────────────────────────────────────────
# Narrate-then-act detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that suggest the model is about to do something but hasn't yet
NARRATE_THEN_ACT_PATTERNS = [
    r"\bI'?ll\s+\w+",              # "I'll search for..."
    r"\bI'?m\s+going\s+to\s+\w+",  # "I'm going to look..."
    r"\bLet\s+me\s+\w+",           # "Let me check..."
    r"\bNow\s+I\s+(will|'ll)",     # "Now I will..."
    r"\bI\s+need\s+to\s+\w+",      # "I need to search..."
    r"\bNext,?\s+I",               # "Next, I will..."
    r"\bFirst,?\s+(I|let)",        # "First, I will..." / "First, let me..."
    r"\bI\s+should\s+\w+",         # "I should check..."
    r"\bI\s+will\s+\w+",           # "I will search..."
    r"\bGoing\s+to\s+\w+",         # "Going to check..."
    r"\bMy\s+plan\s+is\s+to",      # "My plan is to..."
]

# Completion signals - if these appear, model is genuinely done (not a false finish)
COMPLETION_SIGNALS = [
    r"\bFinal\s+Answer\b",
    r"\bThe\s+answer\s+is\b",
    r"\bIn\s+conclusion\b",
    r"\bTo\s+summarize\b",
    r"\bIn\s+summary\b",
    r"\b✅\b",
    r"\bhas\s+been\s+(fully\s+)?answered\b",
    r"\bquestion\s+has\s+been\s+(fully\s+)?(answered|solved|completed)\b",
    r"\btask\s+is\s+complete\b",
    r"\bsolution\s+is\s+complete\b",
]

NARRATE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in NARRATE_THEN_ACT_PATTERNS]
COMPLETION_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in COMPLETION_SIGNALS]


def _has_completion_signal(text: str) -> bool:
    """Check if text contains signals that the model believes it's done."""
    for pattern in COMPLETION_PATTERNS_COMPILED:
        if pattern.search(text):
            return True
    return False


def _detect_narrate_then_act(text: str) -> tuple[bool, list[str]]:
    """
    Detect if the text contains narrate-then-act signals.
    Returns (detected, matched_phrases).

    Key insight: Only flag patterns at the END of responses.
    Claude often says "Let me..." then immediately does it in the same response.
    GPT says "I'll..." at the end and stops.

    Position-aware detection prevents false positives for Claude while still
    catching GPT's premature exits.
    """
    if not text:
        return False, []

    # Key insight: if completion signals present, model is done - not a false finish
    if _has_completion_signal(text):
        return False, []

    # Only check the TRAILING portion of the response
    # If narration appears early followed by substantial content, it's fine
    trailing_window = 300  # chars from end to check
    min_content_after = 100  # chars after pattern to consider it "followed by content"

    trailing_text = text[-trailing_window:] if len(text) > trailing_window else text

    matched = []
    for pattern in NARRATE_PATTERNS_COMPILED:
        for match in pattern.finditer(trailing_text):
            # Calculate position relative to end of full text
            match_end_from_text_end = len(trailing_text) - match.end()

            # Only flag if pattern is near the very end with little content after
            if match_end_from_text_end < min_content_after:
                matched.append(match.group(0))

    return len(matched) > 0, matched[:3]


def _get_escalating_nudge(attempt: int, matched_phrases: list[str]) -> str:
    """
    Generate an escalating nudge based on consecutive false finishes.

    Level 1: Gentle reminder
    Level 2: Quote what they said, ask them to act
    Level 3: Direct instruction to act or finish
    """
    phrase = matched_phrases[0] if matched_phrases else "you would take an action"

    if attempt == 1:
        return "Continue with your plan."
    elif attempt == 2:
        return f'You said "{phrase}" but didn\'t call any tools. Please execute the action now.'
    else:
        return (
            "IMPORTANT: You must either call a tool to take action, or provide your "
            "final answer. Do not describe what you will do - just do it."
        )


def run(
    prompt: str,
    provider: Provider = "anthropic",
    model: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_iterations: int = 20,
    max_consecutive_false_finishes: int = 3,
    tools_list: list | None = None,
    tool_functions: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the agent loop with adaptive finish detection.

    This harness detects when the model says "I'll do X" without actually doing it
    (narrate-then-act pattern) and nudges it to continue. It also recognizes genuine
    completion signals to avoid false positives.

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
    false_finish_streak = 0
    total_false_finishes = 0
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
            "false_finish": False,
            "narrate_then_act": False,
            "matched_phrases": [],
            "nudge": None,
        }

        if verbose:
            console.print(Rule(Text(f" Iteration {i} ", style="bold cyan"), style="cyan"))

        response: ModelResponse = call_model(
            provider=provider,
            model=model,
            messages=messages,
            tools=_tools_list,
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
        # NO TOOL CALLS: Check if genuine completion or false finish
        # ─────────────────────────────────────────────────────────────
        if not response.tool_calls:
            is_false_finish, matched_phrases = _detect_narrate_then_act(response.text)

            if is_false_finish:
                false_finish_streak += 1
                total_false_finishes += 1
                iter_data["false_finish"] = True
                iter_data["narrate_then_act"] = True
                iter_data["matched_phrases"] = matched_phrases

                if verbose:
                    console.print(Text("🔍 FALSE-FINISH DETECTED:", style="bold magenta"))
                    console.print(f"   [dim]Matched phrases:[/dim] {matched_phrases}")
                    console.print(f"   [dim]Consecutive:[/dim] {false_finish_streak}/{max_consecutive_false_finishes}")

                # Safety: too many consecutive false finishes = stuck
                if false_finish_streak >= max_consecutive_false_finishes:
                    finished_reason = "stuck_narration_loop"
                    if verbose:
                        console.print(Text("   ⚠️ Max consecutive false-finishes reached.", style="bold red"))
                    iterations.append(iter_data)
                    break

                # Escalating nudge
                nudge = _get_escalating_nudge(false_finish_streak, matched_phrases)
                iter_data["nudge"] = nudge
                if verbose:
                    console.print(Text(f"   💬 NUDGE ({false_finish_streak}/3): {nudge}", style="bold magenta"))

                messages.append(format_assistant_message(provider, response))
                messages.append({"role": "user", "content": nudge})
                iterations.append(iter_data)
                continue

            else:
                # Genuine completion (no narrate-then-act, or has completion signals)
                if verbose:
                    console.print(Text("✅ EXIT: Genuine completion", style="bold green"))
                iter_data["finish_triggered"] = True
                finished_reason = "genuine_completion"
                iterations.append(iter_data)
                break

        # ─────────────────────────────────────────────────────────────
        # HAS TOOL CALLS: Reset false finish streak, execute tools
        # ─────────────────────────────────────────────────────────────
        if false_finish_streak > 0 and verbose:
            console.print(Text(f"   [dim]Reset false-finish counter (was {false_finish_streak})[/dim]"))
        false_finish_streak = 0

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
            f"[bold]False finishes:[/bold] {total_false_finishes}\n"
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
        "false_finishes": total_false_finishes,
        "narrate_then_act": total_false_finishes,  # Same metric
        "task_complete": finished_reason == "genuine_completion",
        "finished_reason": finished_reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Finish Harness"
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
        "--max-false-finishes",
        type=int,
        default=3,
        help="Max consecutive false-finishes before force-exit (default: 3)",
    )

    args = parser.parse_args()

    result = run(
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
        max_consecutive_false_finishes=args.max_false_finishes,
    )

    return result


if __name__ == "__main__":
    main()
