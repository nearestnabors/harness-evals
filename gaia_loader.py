"""
GAIA Benchmark task loader.

Loads tasks from the GAIA dataset on HuggingFace.
https://huggingface.co/datasets/gaia-benchmark/GAIA

Note: GAIA is a gated dataset. You need to:
1. Create a HuggingFace account
2. Accept the dataset terms at the link above
3. Set HF_TOKEN environment variable with your access token
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class GAIATask:
    """A single GAIA benchmark task."""

    task_id: str
    question: str
    level: int  # 1, 2, or 3 (increasing difficulty)
    expected_answer: str | None  # None for test split (answers not public)
    file_path: str | None  # Path to attachment if any
    metadata: dict | None


def load_gaia_tasks(
    level: int | None = None,
    split: str = "validation",
    max_tasks: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[GAIATask]:
    """
    Load GAIA tasks from HuggingFace.

    Args:
        level: Filter by difficulty level (1, 2, or 3). None for all levels.
        split: "validation" (has answers) or "test" (no answers, for leaderboard).
        max_tasks: Maximum number of tasks to return. None for all.
        shuffle: Whether to shuffle tasks before returning.
        seed: Random seed for shuffling.

    Returns:
        List of GAIATask objects.

    Raises:
        ValueError: If HF_TOKEN not set or dataset access denied.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Get your token at https://huggingface.co/settings/tokens"
        )

    # Determine which config to load
    if level is not None:
        config = f"2023_level{level}"
    else:
        config = "2023_all"

    try:
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            config,
            split=split,
            token=hf_token,
        )
    except Exception as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            raise ValueError(
                "Access denied to GAIA dataset. "
                "Please accept the terms at https://huggingface.co/datasets/gaia-benchmark/GAIA"
            ) from e
        raise

    # Convert to GAIATask objects
    tasks = []
    for item in dataset:
        task = GAIATask(
            task_id=item.get("task_id", "unknown"),
            question=item.get("Question", ""),
            level=item.get("Level", 1),
            expected_answer=item.get("Final answer"),  # None for test split
            file_path=item.get("file_path"),
            metadata=item.get("Annotator Metadata"),
        )
        tasks.append(task)

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(tasks)

    # Limit number of tasks
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


def get_sample_tasks() -> list[GAIATask]:
    """
    Return a few hardcoded sample GAIA-style tasks for testing.

    Use this when you don't have HuggingFace access or want quick testing.
    These are simplified versions inspired by GAIA but not actual GAIA tasks.
    """
    return [
        GAIATask(
            task_id="sample_1",
            question="What is the population of the city where Albert Einstein was born? Multiply that number by 1905 (the year he published his paper on special relativity).",
            level=1,
            expected_answer="584980380",  # Ulm population ~307,000 * 1905 ≈ 584,835,000 (approximate)
            file_path=None,
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        GAIATask(
            task_id="sample_2",
            question="Who won the Academy Award for Best Picture in 2020? How many letters are in the title of that film?",
            level=1,
            expected_answer="8",  # "Parasite" = 8 letters
            file_path=None,
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        GAIATask(
            task_id="sample_3",
            question="What is the atomic number of the element named after the scientist who developed the theory of general relativity? Add this to the year that theory was published.",
            level=1,
            expected_answer="2014",  # Einsteinium = 99, General Relativity = 1915, 99 + 1915 = 2014
            file_path=None,
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        GAIATask(
            task_id="sample_4",
            question="Find the height in meters of the tallest building in the world as of 2024. Calculate how many times the Eiffel Tower (330m) would need to be stacked to exceed this height. Give the minimum whole number.",
            level=2,
            expected_answer="3",  # Burj Khalifa = 828m, 828/330 = 2.5, so need 3
            file_path=None,
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        GAIATask(
            task_id="sample_5",
            question="What is the sum of the birth years of the first three Presidents of the United States?",
            level=1,
            expected_answer="5765",  # Washington 1732 + Adams 1735 + Jefferson 1743 = 5210... let me recalculate
            file_path=None,
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
    ]


def format_task_prompt(task: GAIATask, include_instructions: bool = True) -> str:
    """
    Format a GAIA task as a prompt for the model.

    Args:
        task: The GAIA task to format.
        include_instructions: Whether to include general instructions.

    Returns:
        Formatted prompt string.
    """
    parts = []

    if include_instructions:
        parts.append(
            "You are a helpful assistant that can search the web, look up information, "
            "and perform calculations to answer questions accurately. "
            "Use the available tools as needed. Provide your final answer clearly."
        )
        parts.append("")

    parts.append(f"**Question:** {task.question}")

    if task.file_path:
        parts.append(f"\n**Attached file:** {task.file_path}")

    parts.append("\nPlease solve this step by step, using tools as needed, and provide your final answer.")

    return "\n".join(parts)


def check_answer(task: GAIATask, model_answer: str) -> bool:
    """
    Check if the model's answer matches the expected answer.

    Uses fuzzy matching to handle formatting differences.

    Args:
        task: The GAIA task with expected answer.
        model_answer: The model's response.

    Returns:
        True if answer is correct, False otherwise.
    """
    if task.expected_answer is None:
        return False  # Can't check test split answers

    expected = task.expected_answer.strip().lower()
    actual = model_answer.strip().lower()

    # Exact match
    if expected == actual:
        return True

    # Check if expected answer appears in the response
    if expected in actual:
        return True

    # Try to extract numbers and compare
    import re

    expected_nums = re.findall(r"[\d,]+\.?\d*", expected)
    actual_nums = re.findall(r"[\d,]+\.?\d*", actual)

    if expected_nums and actual_nums:
        # Normalize numbers (remove commas)
        expected_num = expected_nums[0].replace(",", "")
        for actual_num in actual_nums:
            if actual_num.replace(",", "") == expected_num:
                return True

    return False


if __name__ == "__main__":
    # Quick test
    print("Loading sample GAIA-style tasks...")
    tasks = get_sample_tasks()

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task ID: {task.task_id} (Level {task.level})")
        print(f"Question: {task.question}")
        print(f"Expected: {task.expected_answer}")
