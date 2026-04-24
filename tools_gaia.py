"""
Real tools for GAIA benchmark tasks.

These tools actually call APIs and perform real operations:
- web_search: Search the web using Tavily API
- calculator: Evaluate mathematical expressions
- read_file: Read content from files (for GAIA attachments)
- wikipedia_lookup: Get Wikipedia article summaries
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any

# Tool definitions for the LLM (Anthropic/OpenAI format)
tools_list = [
    {
        "name": "web_search",
        "description": "Search the web for information. Returns relevant snippets from web pages. Use this for current events, facts, or any information you need to look up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **), parentheses, and common math functions (sqrt, sin, cos, tan, log, exp, abs, round, floor, ceil). Use this for any calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '(42 * 1905) + sqrt(144)')",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this to read attached documents, CSVs, or text files provided with the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "wikipedia_lookup",
        "description": "Look up a Wikipedia article and get a summary. Use this for factual information about people, places, events, concepts, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to look up on Wikipedia",
                },
            },
            "required": ["topic"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API.

    Requires TAVILY_API_KEY environment variable.
    Get a free API key at https://tavily.com
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not set. Get a free key at https://tavily.com"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )

        # Format results
        results = []

        # Include the AI-generated answer if available
        if response.get("answer"):
            results.append(f"**Summary:** {response['answer']}\n")

        # Include search results
        for i, result in enumerate(response.get("results", [])[:max_results], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            results.append(f"{i}. **{title}**\n   URL: {url}\n   {content}\n")

        if not results:
            return "No results found for this query."

        return "\n".join(results)

    except Exception as e:
        return f"Error performing web search: {e}"


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Supports: +, -, *, /, **, (), and math functions.
    """
    # Define allowed names for safe evaluation
    allowed_names = {
        # Math constants
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        # Math functions
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        # From math module
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    try:
        # Remove any potentially dangerous characters
        # Allow: digits, operators, parentheses, dots, commas, spaces, function names
        sanitized = expression.strip()

        # Check for obviously dangerous patterns
        dangerous_patterns = [
            r"__",  # Dunder methods
            r"import",
            r"exec",
            r"eval",
            r"open",
            r"file",
            r"input",
            r"print",
            r"\[",  # List indexing (could access __builtins__)
            r"\]",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                return f"Error: Expression contains disallowed pattern: {pattern}"

        # Evaluate with restricted namespace
        result = eval(sanitized, {"__builtins__": {}}, allowed_names)

        # Format result nicely
        if isinstance(result, float):
            # Avoid scientific notation for reasonable numbers
            if abs(result) < 1e10 and abs(result) > 1e-10 or result == 0:
                # Round to avoid floating point artifacts
                if result == int(result):
                    return str(int(result))
                return str(round(result, 10))
            else:
                return str(result)

        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {e}"
    except SyntaxError as e:
        return f"Error: Invalid syntax - {e}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


def read_file(file_path: str) -> str:
    """
    Read content from a file.

    Supports: .txt, .csv, .json, .md, .py, and other text files.
    For binary files like PDFs, returns an error suggesting web search.
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File not found: {file_path}"

        # Check file extension
        suffix = path.suffix.lower()

        # Binary files we can't read directly
        binary_extensions = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".mp3", ".mp4"}

        if suffix in binary_extensions:
            return f"Error: Cannot read binary file ({suffix}). Try searching for the content online or describe what you need from this file."

        # Read text files
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Truncate if very long
        max_chars = 50000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... [truncated, {len(content) - max_chars} more characters]"

        return content

    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


def wikipedia_lookup(topic: str) -> str:
    """
    Look up a topic on Wikipedia using the API.

    Returns the article summary/extract.
    """
    import urllib.request
    import urllib.parse

    try:
        # Use Wikipedia API to get article summary
        base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        encoded_topic = urllib.parse.quote(topic.replace(" ", "_"))
        url = base_url + encoded_topic

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "HarnessEvals/1.0 (research project)"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        title = data.get("title", topic)
        extract = data.get("extract", "No summary available.")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        result = f"**{title}**\n\n{extract}"
        if page_url:
            result += f"\n\nSource: {page_url}"

        return result

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"No Wikipedia article found for '{topic}'. Try a different search term or use web_search."
        return f"Error looking up Wikipedia: HTTP {e.code}"
    except Exception as e:
        return f"Error looking up Wikipedia: {e}"


# Map tool names to functions
tool_functions = {
    "web_search": web_search,
    "calculator": calculator,
    "read_file": read_file,
    "wikipedia_lookup": wikipedia_lookup,
}


def reset_tasks():
    """Reset function for compatibility with harnesses. No-op for GAIA tools."""
    pass


# For compatibility with existing harnesses
TASKS = []
