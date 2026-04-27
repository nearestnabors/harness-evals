"""
Harness C: Adaptive finish (hybrid)

Finish condition: Smart detection of genuine completion vs. false finishes.
- Detects "narrate-then-act" patterns ("I'll search for..." without calling tools)
- Recognizes completion signals ("Final Answer:", checkmarks, etc.)
- Uses escalating nudges when model appears stuck
"""

import re
from models import call_model, format_assistant_message, format_tool_result_message, DEFAULT_MODELS


# Patterns that suggest the model is about to do something (narrate-then-act)
NARRATE_PATTERNS = [
    r"\bI'?ll\s+\w+",              # "I'll search for..."
    r"\bLet\s+me\s+\w+",           # "Let me check..."
    r"\bI'?m\s+going\s+to\s+\w+",  # "I'm going to look..."
    r"\bNow\s+I\s+(will|'ll)",     # "Now I will..."
    r"\bI\s+need\s+to\s+\w+",      # "I need to search..."
    r"\bI\s+will\s+\w+",           # "I will search..."
]

# Completion signals - if present, model is done (not a false finish)
COMPLETION_SIGNALS = [
    r"\bFinal\s+Answer\b",
    r"\bThe\s+answer\s+is\b",
    r"\bIn\s+conclusion\b",
    r"\b✅\b",
    r"\btask\s+is\s+complete\b",
]

NARRATE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NARRATE_PATTERNS]
COMPLETION_COMPILED = [re.compile(p, re.IGNORECASE) for p in COMPLETION_SIGNALS]


def _has_completion_signal(text: str) -> bool:
    """Check if text contains signals that the model is done."""
    return any(p.search(text) for p in COMPLETION_COMPILED)


def _detect_narrate_then_act(text: str) -> tuple[bool, list[str]]:
    """Detect if text contains narrate-then-act patterns."""
    if not text or _has_completion_signal(text):
        return False, []

    matched = []
    for pattern in NARRATE_COMPILED:
        for match in pattern.finditer(text):
            matched.append(match.group(0))

    return len(matched) > 0, matched[:3]


def _get_escalating_nudge(count: int, phrases: list[str]) -> str:
    """Generate escalating nudge based on consecutive false finishes."""
    phrase = phrases[0] if phrases else "you would take an action"

    if count == 1:
        return "Continue with your plan."
    elif count == 2:
        return f'You said "{phrase}" but didn\'t call any tools. Please execute the action now.'
    else:
        return "IMPORTANT: Call a tool or provide your final answer. Do not describe what you will do."


def run(
    prompt: str,
    provider: str = "anthropic",
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 20,
    max_false_finishes: int = 3,
    tools_list: list | None = None,
    tool_functions: dict | None = None,
) -> dict:
    """
    Run the agent loop with adaptive finish detection.

    Returns a clean result dict with iteration details for optional tracing.
    """
    model = model or DEFAULT_MODELS[provider]
    tools_list = tools_list or []
    tool_functions = tool_functions or {}

    messages = [{"role": "user", "content": prompt}]
    iterations = []
    all_text = ""
    false_finish_streak = 0
    total_false_finishes = 0
    finished_reason = "max_iterations"

    for i in range(1, max_iterations + 1):
        iter_data = {
            "n": i,
            "tool_calls": [],
            "text": "",
            "finish_triggered": False,
            "false_finish": False,
            "narrate_then_act": False,
            "nudge": None,
        }

        # Call the model
        response = call_model(
            provider=provider,
            model=model,
            messages=messages,
            tools=tools_list,
            system_prompt=system_prompt,
        )

        iter_data["input_tokens"] = response.tokens.input_tokens
        iter_data["output_tokens"] = response.tokens.output_tokens

        if response.text:
            iter_data["text"] = response.text
            all_text += response.text + "\n"

        # No tool calls - check if genuine completion or false finish
        if not response.tool_calls:
            is_false_finish, matched = _detect_narrate_then_act(response.text)

            if is_false_finish:
                false_finish_streak += 1
                total_false_finishes += 1
                iter_data["false_finish"] = True
                iter_data["narrate_then_act"] = True
                iter_data["matched_phrases"] = matched

                # Too many consecutive false finishes = stuck
                if false_finish_streak >= max_false_finishes:
                    finished_reason = "stuck_narration_loop"
                    iterations.append(iter_data)
                    break

                # Nudge the model to continue
                nudge = _get_escalating_nudge(false_finish_streak, matched)
                iter_data["nudge"] = nudge
                messages.append(format_assistant_message(provider, response))
                messages.append({"role": "user", "content": nudge})
                iterations.append(iter_data)
                continue

            else:
                # Genuine completion
                iter_data["finish_triggered"] = True
                finished_reason = "genuine_completion"
                iterations.append(iter_data)
                break

        # Has tool calls - reset false finish streak
        false_finish_streak = 0
        messages.append(format_assistant_message(provider, response))

        for tool_call in response.tool_calls:
            name = tool_call.name
            args = tool_call.arguments

            if name in tool_functions:
                try:
                    result = tool_functions[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {name}"

            iter_data["tool_calls"].append({
                "name": name,
                "arguments": args,
                "result": str(result)[:500],
            })

            messages.append(
                format_tool_result_message(provider, tool_call.id, name, str(result))
            )

        iterations.append(iter_data)

    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "iteration_count": len(iterations),
        "tool_calls_total": sum(len(it["tool_calls"]) for it in iterations),
        "input_tokens": sum(it["input_tokens"] for it in iterations),
        "output_tokens": sum(it["output_tokens"] for it in iterations),
        "false_finishes": total_false_finishes,
        "finished_reason": finished_reason,
        "task_complete": finished_reason == "genuine_completion",
    }
