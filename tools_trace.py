"""
Shared tool definitions for the finish-condition eval harnesses.
Extracted from Jose's agent-poc.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# -----------------------------
# Module-level TODO List Store
# -----------------------------
# Keeps tasks in-memory across tool calls within the same process.
# Each task is a dict with keys: id, description, deliverable, data, status
TASKS: List[Dict[str, str]] = []


def _generate_next_task_id() -> str:
    """Generate the next numeric id as a string based on existing ids."""
    max_id_numeric = 0
    for task in TASKS:
        task_id = task.get("id", "").strip()
        try:
            max_id_numeric = max(max_id_numeric, int(task_id))
        except ValueError:
            continue
    return str(max_id_numeric + 1)


def reset_tasks():
    """Clear all tasks. Useful for resetting state between eval runs."""
    global TASKS
    TASKS = []


def todo_write(
    description: str,
    deliverable: str,
    id: Optional[str] = None,
    data: Optional[str] = None,
    status: Optional[str] = None,
):
    """
    Write (append or update) a task in the in-memory task list.

    - If `id` is omitted, a new numeric id is generated and the task is appended.
    - If `id` is provided and exists, the task is updated in place.

    Returns the created/updated task.
    """
    if not description or not deliverable:
        return {"error": "Both 'description' and 'deliverable' are required"}

    task_id = (id or _generate_next_task_id()).strip()

    # Try update if id exists
    for task in TASKS:
        if task.get("id") == task_id:
            task["description"] = description
            task["deliverable"] = deliverable
            if data is not None:
                task["data"] = data
            if status is not None:
                task["status"] = status
            return task

    # Append new task
    new_task: Dict[str, str] = {
        "id": task_id,
        "description": description,
        "deliverable": deliverable,
        "data": data if data is not None else "N/A",
        "status": status if status is not None else "pending",
    }
    TASKS.append(new_task)
    return new_task


def todo_read():
    """Return the current list of tasks (list of dicts)."""
    return TASKS


def calculator(expression):
    """Evaluate a mathematical expression."""
    # Remove any non-digit or non-operator characters from the expression
    expression = re.sub(r"[^0-9+\-*/().]", "", expression)

    try:
        result = eval(expression)
        return str(result)
    except (SyntaxError, ZeroDivisionError, NameError, TypeError, OverflowError):
        return "Error: Invalid expression"


# Path to sample trace data (relative to this file)
TRACE_CSV_PATH = Path(__file__).resolve().parent / "sample_trace.csv"


def get_trace_preview():
    """Get a preview of a complete trace. All fields truncated to 30 chars."""
    df = pd.read_csv(TRACE_CSV_PATH)
    df = df.set_index("id")
    df.drop(
        columns=["attributes.input.value", "attributes.output.value"],
        inplace=True,
        errors="ignore",
    )
    df = df.apply(
        lambda col: col.map(lambda val: val[:30] if isinstance(val, str) else val)
    )
    df.dropna(inplace=True, axis=1)
    return df.to_dict(orient="records")


def get_span_data(ids: list[str]):
    """Get all data for a single or list of spans by their IDs."""
    df = pd.read_csv(TRACE_CSV_PATH)
    df = df.set_index("id")
    df = df.loc[ids]
    return df.to_dict(orient="records")


def find_in_trace(query: str, max_results: int = 20):
    """
    Perform a free-text search over the entire trace (all columns).
    Mimics a spreadsheet "Ctrl+F" style search.

    Args:
        query: Search query string. Space-separated terms are OR'd.
        max_results: Maximum number of matches to return.

    Returns:
        List of dicts with keys: id, column, content
    """
    df = pd.read_csv(TRACE_CSV_PATH)

    terms = [re.escape(term) for term in query.split() if term.strip()]
    if not terms:
        return []
    pattern = re.compile("|".join(terms), re.IGNORECASE)

    matches: list[dict] = []

    for row_idx, row in df.iterrows():
        row_id = row.get("id", row_idx)
        for col, value in row.items():
            if pd.isna(value):
                continue
            value_str = str(value)
            if pattern.search(value_str):
                matches.append(
                    {
                        "id": row_id,
                        "column": col,
                        "content": value_str,
                    }
                )
            if len(matches) >= max_results:
                return matches

    return matches


# -----------------------------
# Tool Definitions (schemas)
# -----------------------------

tools_list = [
    {
        "name": "calculator",
        "description": "A simple calculator that performs basic arithmetic operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4').",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_trace_preview",
        "description": "Get a preview of a complete trace. All the fields are truncated to 50 chars.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_span_data",
        "description": "Get the all the data for a single or a list of spans. The ids are the span ids from the trace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "find_in_trace",
        "description": "Search the trace for a query string and return matching cells.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Multiple terms can be separated by spaces (OR logic).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "todo_write",
        "description": "Append or update a task in the in-memory task list (list of dicts). If id is omitted, a new numeric id is generated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Optional explicit task id."},
                "description": {"type": "string", "description": "Task description."},
                "deliverable": {
                    "type": "string",
                    "description": "Expected deliverable of the task.",
                },
                "data": {
                    "type": "string",
                    "description": "Optional data payload for the task.",
                },
                "status": {
                    "type": "string",
                    "description": "Task status (default: 'pending').",
                },
            },
            "required": ["description", "deliverable"],
        },
    },
    {
        "name": "todo_read",
        "description": "Read the current list of tasks (list of dicts).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

# Map tool names to functions for dynamic invocation
tool_functions = {
    "calculator": calculator,
    "get_trace_preview": get_trace_preview,
    "get_span_data": get_span_data,
    "find_in_trace": find_in_trace,
    "todo_write": todo_write,
    "todo_read": todo_read,
}
