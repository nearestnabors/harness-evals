"""
Multi-step evaluation tasks for testing finish conditions.

These tasks require tool use (web search, calculator) and multi-step reasoning,
making them effective for revealing differences between finish condition strategies.
"""

from dataclasses import dataclass


@dataclass
class EvalTask:
    """A single evaluation task."""

    task_id: str
    question: str
    level: int  # 1 = simple, 2 = medium, 3 = complex
    expected_answer: str | None
    file_path: str | None = None
    metadata: dict | None = None


def get_sample_tasks() -> list[EvalTask]:
    """
    Return multi-step evaluation tasks.

    These tasks require web search and calculation, exercising the full
    tool-use loop where finish condition strategies matter most.
    """
    return [
        EvalTask(
            task_id="multi_step_1",
            question="What is the population of the city where Albert Einstein was born? Multiply that number by 1905 (the year he published his paper on special relativity).",
            level=1,
            expected_answer="584980380",  # Ulm ~307,000 * 1905
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        EvalTask(
            task_id="multi_step_2",
            question="Who won the Academy Award for Best Picture in 2020? How many letters are in the title of that film?",
            level=1,
            expected_answer="8",  # "Parasite" = 8 letters
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        EvalTask(
            task_id="multi_step_3",
            question="What is the atomic number of the element named after the scientist who developed the theory of general relativity? Add this to the year that theory was published.",
            level=1,
            expected_answer="2014",  # Einsteinium = 99, 1915, 99 + 1915 = 2014
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        EvalTask(
            task_id="multi_step_4",
            question="Find the height in meters of the tallest building in the world as of 2024. Calculate how many times the Eiffel Tower (330m) would need to be stacked to exceed this height. Give the minimum whole number.",
            level=2,
            expected_answer="3",  # Burj Khalifa = 828m, 828/330 = 2.5, so need 3
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        EvalTask(
            task_id="multi_step_5",
            question="What is the sum of the birth years of the first three Presidents of the United States?",
            level=1,
            expected_answer="5210",  # Washington 1732 + Adams 1735 + Jefferson 1743
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        # Tasks designed to trigger "narrate-then-act" false finishes
        EvalTask(
            task_id="planning_1",
            question="I need you to think through this carefully. First, consider what information you'll need, then outline your approach, and finally execute it: What is the current population of the capital city of the country that hosted the 2024 Summer Olympics, divided by 1000?",
            level=2,
            expected_answer="2102",  # Paris ~2.1M, /1000 = ~2102
            metadata={"tools_needed": ["web_search", "calculator"], "tests_false_finish": True},
        ),
        EvalTask(
            task_id="planning_2",
            question="Let me explain what I need: I want to know the atomic number of gold. I think it's around 79, but please verify this and then multiply it by 3. Walk me through your reasoning first.",
            level=1,
            expected_answer="237",  # Gold = 79, 79 * 3 = 237
            metadata={"tools_needed": ["web_search", "calculator"], "tests_false_finish": True},
        ),
        EvalTask(
            task_id="planning_3",
            question="Before you answer, plan out the steps you'll take. Then: Find the year the Eiffel Tower was completed, subtract 1000, and tell me what happened that year in European history.",
            level=2,
            expected_answer="889",  # Eiffel Tower 1889 - 1000 = 889
            metadata={"tools_needed": ["web_search", "calculator"], "tests_false_finish": True},
        ),
    ]


def format_task_prompt(task: EvalTask, include_instructions: bool = True) -> str:
    """Format a task as a prompt for the model."""
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


def check_answer(task: EvalTask, model_answer: str) -> bool:
    """
    Check if the model's answer matches the expected answer.
    Uses fuzzy matching to handle formatting differences.
    """
    if task.expected_answer is None:
        return False

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
        expected_num = expected_nums[0].replace(",", "")
        for actual_num in actual_nums:
            if actual_num.replace(",", "") == expected_num:
                return True

    return False


if __name__ == "__main__":
    print("Multi-step evaluation tasks:")
    tasks = get_sample_tasks()

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task ID: {task.task_id} (Level {task.level})")
        print(f"Question: {task.question}")
        print(f"Expected: {task.expected_answer}")
