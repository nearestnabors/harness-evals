"""
Model provider abstraction for Anthropic and OpenAI.
Returns responses in a common format: { text: str, tool_calls: list, tokens: TokenUsage }
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")


@dataclass
class ToolCall:
    """Normalized tool call representation."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    """Token usage from a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResponse:
    """Common response format across providers."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    raw_response: Any = None  # Original response for debugging


Provider = Literal["anthropic", "openai", "openrouter"]


def get_client(provider: Provider):
    """Get the appropriate client for the provider."""
    if provider == "anthropic":
        from anthropic import Anthropic

        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    elif provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider == "openrouter":
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _convert_tools_for_anthropic(tools: list[dict]) -> list[dict]:
    """Convert tool definitions to Anthropic format."""
    from anthropic.types import ToolParam

    return [
        ToolParam(
            name=tool["name"],
            description=tool.get("description", ""),
            input_schema=tool.get("input_schema", {}),
        )
        for tool in tools
    ]


def _convert_tools_for_openai(tools: list[dict]) -> list[dict]:
    """Convert tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        for tool in tools
    ]


def _parse_anthropic_response(response) -> ModelResponse:
    """Parse Anthropic response into common format."""
    from anthropic.types import TextBlock, ToolUseBlock

    text_parts = []
    tool_calls = []

    for content in response.content:
        if isinstance(content, TextBlock):
            text_parts.append(content.text)
        elif isinstance(content, ToolUseBlock):
            tool_calls.append(
                ToolCall(
                    id=content.id,
                    name=content.name,
                    arguments=content.input if isinstance(content.input, dict) else {},
                )
            )

    # Extract token usage
    tokens = TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    return ModelResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        tokens=tokens,
        raw_response=response,
    )


def _parse_openai_response(response) -> ModelResponse:
    """Parse OpenAI response into common format."""
    import json

    message = response.choices[0].message
    text = message.content or ""
    tool_calls = []

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )

    # Extract token usage
    tokens = TokenUsage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )

    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        tokens=tokens,
        raw_response=response,
    )


def call_model(
    provider: Provider,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    system_prompt: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> ModelResponse:
    """
    Call a model with the given messages and tools.

    Args:
        provider: "anthropic" or "openai"
        model: Model name (e.g., "claude-sonnet-4-20250514" or "gpt-4o")
        messages: List of message dicts with "role" and "content"
        tools: Optional list of tool definitions
        system_prompt: System prompt string
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        ModelResponse with text, tool_calls, tokens, and raw_response
    """
    client = get_client(provider)

    if provider == "anthropic":
        # Anthropic uses system as a separate parameter
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = _convert_tools_for_anthropic(tools)

        response = client.messages.create(**kwargs)
        return _parse_anthropic_response(response)

    elif provider in ("openai", "openrouter"):
        # OpenAI and OpenRouter use same API format
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.extend(messages)

        kwargs = {
            "model": model,
            "temperature": temperature,
            "messages": openai_messages,
        }
        # OpenRouter uses max_tokens, OpenAI uses max_completion_tokens
        if provider == "openrouter":
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_completion_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = _convert_tools_for_openai(tools)

        response = client.chat.completions.create(**kwargs)
        return _parse_openai_response(response)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def format_tool_result_message(
    provider: Provider,
    tool_call_id: str,
    tool_name: str,
    result: str,
) -> dict:
    """Format a tool result message for the given provider."""
    if provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result,
                }
            ],
        }
    elif provider in ("openai", "openrouter"):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")


def format_assistant_message(provider: Provider, response: ModelResponse) -> dict:
    """Format an assistant message to add to conversation history."""
    if provider == "anthropic":
        # Anthropic expects the raw content blocks
        return {"role": "assistant", "content": response.raw_response.content}
    elif provider in ("openai", "openrouter"):
        # OpenAI/OpenRouter expects the message object
        return response.raw_response.choices[0].message.model_dump()
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Default models for each provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "openrouter": "google/gemma-4-31b-it",  # Paid tier (free tier has upstream limits)
}

# Models to use in eval matrix
EVAL_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "openrouter": "google/gemma-4-31b-it",  # Paid tier (free tier has upstream limits)
}
