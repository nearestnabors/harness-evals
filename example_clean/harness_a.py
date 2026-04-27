"""
Harness A: Implicit finish (Claude Code style)

Finish condition: The model stops calling tools (no tool_calls in response).
This is the simplest approach - the model decides when it's done.
"""

from models import call_model, format_assistant_message, format_tool_result_message, DEFAULT_MODELS


def run(
    prompt: str,
    provider: str = "anthropic",
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 20,
    tools_list: list | None = None,
    tool_functions: dict | None = None,
) -> dict:
    """
    Run the agent loop with implicit finish condition.

    Returns a clean result dict with iteration details for optional tracing.
    """
    model = model or DEFAULT_MODELS[provider]
    tools_list = tools_list or []
    tool_functions = tool_functions or {}

    messages = [{"role": "user", "content": prompt}]
    iterations = []
    all_text = ""
    finished_reason = "max_iterations"

    for i in range(1, max_iterations + 1):
        # Track this iteration
        iter_data = {
            "n": i,
            "tool_calls": [],
            "text": "",
            "finish_triggered": False,
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

        # Capture any text
        if response.text:
            iter_data["text"] = response.text
            all_text += response.text + "\n"

        # No tool calls = we're done
        if not response.tool_calls:
            iter_data["finish_triggered"] = True
            finished_reason = "no_tool_calls"
            iterations.append(iter_data)
            break

        # Execute tool calls
        messages.append(format_assistant_message(provider, response))

        for tool_call in response.tool_calls:
            name = tool_call.name
            args = tool_call.arguments

            # Execute
            if name in tool_functions:
                try:
                    result = tool_functions[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {name}"

            # Track it
            iter_data["tool_calls"].append({
                "name": name,
                "arguments": args,
                "result": str(result)[:500],  # Truncate for storage
            })

            # Add to messages
            messages.append(
                format_tool_result_message(provider, tool_call.id, name, str(result))
            )

        iterations.append(iter_data)

    # Return clean result with iteration details
    return {
        "text": all_text.strip(),
        "iterations": iterations,
        "iteration_count": len(iterations),
        "tool_calls_total": sum(len(it["tool_calls"]) for it in iterations),
        "input_tokens": sum(it["input_tokens"] for it in iterations),
        "output_tokens": sum(it["output_tokens"] for it in iterations),
        "finished_reason": finished_reason,
        "task_complete": finished_reason == "no_tool_calls",
    }
