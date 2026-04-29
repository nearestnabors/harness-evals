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

    Task categories:
    - Lookup + Calculate: Simple tool chains with verifiable answers
    - Research + Synthestic: Open-ended research requiring judgment
    - File-based: Tasks requiring file reading
    - Iterative: Tasks where initial results may need follow-up
    - Summary-finale: Tasks ending with non-tool work (narrate-then-act test)
    """
    return [
        # ─────────────────────────────────────────────────────────────
        # Lookup + Calculate (3 tasks with verifiable answers)
        # ─────────────────────────────────────────────────────────────
        EvalTask(
            task_id="lookup_calc_1",
            question="What is the population of the city where Albert Einstein was born? Multiply that number by 1905 (the year he published his paper on special relativity).",
            level=1,
            expected_answer="584980380",  # Ulm ~307,000 * 1905
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        EvalTask(
            task_id="lookup_calc_2",
            question="Find the height in meters of the tallest building in the world as of 2024. Calculate how many times the Eiffel Tower (330m) would need to be stacked to exceed this height. Give the minimum whole number.",
            level=2,
            expected_answer="3",  # Burj Khalifa = 828m, 828/330 = 2.5, so need 3
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        EvalTask(
            task_id="lookup_calc_3",
            question="What is the atomic number of the element named after the scientist who developed the theory of general relativity? Add this to the year that theory was published.",
            level=1,
            expected_answer="2014",  # Einsteinium = 99, 1915 + 99 = 2014
            metadata={"tools_needed": ["web_search", "calculator"]},
        ),
        # ─────────────────────────────────────────────────────────────
        # Research + Synthesis (open-ended, no single correct answer)
        # ─────────────────────────────────────────────────────────────
        EvalTask(
            task_id="research_synthesis_1",
            question="""Research the environmental impact of electric vehicles vs gasoline vehicles. Consider manufacturing, operation, and end-of-life. Provide a balanced summary with specific data points.""",
            level=2,
            expected_answer=None,
            metadata={
                "tools_needed": ["web_search"],
                "description": "Open-ended research with no clear stopping point",
            },
        ),
        EvalTask(
            task_id="research_synthesis_2",
            question="""What are the current leading theories about why dinosaurs went extinct? Summarize the evidence for and against each theory.""",
            level=2,
            expected_answer=None,
            metadata={
                "tools_needed": ["web_search", "wikipedia_lookup"],
                "description": "Research task requiring multiple sources and synthesis",
            },
        ),
        # ─────────────────────────────────────────────────────────────
        # File-based tasks (require read_file tool)
        # ─────────────────────────────────────────────────────────────
        EvalTask(
            task_id="file_analysis_1",
            question="""Read the file at ./test_data/sales_data.csv and answer: What was the total revenue for Q1? Which product had the highest sales volume?""",
            level=2,
            expected_answer=None,
            file_path="./test_data/sales_data.csv",
            metadata={
                "tools_needed": ["read_file", "calculator"],
                "description": "File reading + analysis task",
            },
        ),
        EvalTask(
            task_id="file_analysis_2",
            question="""Read the configuration file at ./test_data/config.json and explain what each setting does. Are there any settings that look misconfigured or could cause issues?""",
            level=2,
            expected_answer=None,
            file_path="./test_data/config.json",
            metadata={
                "tools_needed": ["read_file"],
                "description": "File reading + interpretation task",
            },
        ),
        # ─────────────────────────────────────────────────────────────
        # Iterative refinement (may need multiple search rounds)
        # ─────────────────────────────────────────────────────────────
        EvalTask(
            task_id="iterative_1",
            question="""Find three companies currently working on nuclear fusion power. For each, find their most recent funding round and current estimated timeline to commercial operation.""",
            level=2,
            expected_answer=None,
            metadata={
                "tools_needed": ["web_search"],
                "description": "Requires multiple searches to gather complete info",
            },
        ),
        EvalTask(
            task_id="iterative_2",
            question="""What programming language is most in-demand for AI/ML jobs in 2024? Search for recent salary surveys or job posting analyses to support your answer with data.""",
            level=2,
            expected_answer=None,
            metadata={
                "tools_needed": ["web_search"],
                "description": "Research task where first result may need verification",
            },
        ),
        # ─────────────────────────────────────────────────────────────
        # Comparison tasks (structured multi-source research)
        # ─────────────────────────────────────────────────────────────
        EvalTask(
            task_id="comparison_1",
            question="""Compare the specifications of the latest iPhone and Samsung Galaxy flagship phones. Include price, display size, battery capacity, and camera megapixels. Present in a table.""",
            level=2,
            expected_answer=None,
            metadata={
                "tools_needed": ["web_search"],
                "description": "Structured comparison requiring multiple lookups",
            },
        ),
        # ─────────────────────────────────────────────────────────────
        # Summary-finale tasks: expose narrate-then-act pattern
        # These tasks end with synthesis/writing that requires NO tools
        # ─────────────────────────────────────────────────────────────
        EvalTask(
            task_id="summary_finale_1",
            question="""Complete these 15 tasks in order. Use the appropriate tool for each.

1. Look up "Mount Everest" on Wikipedia and note its height in meters
2. Look up "K2" on Wikipedia and note its height in meters
3. Look up "Kangchenjunga" on Wikipedia and note its height in meters
4. Calculate the average height of these three mountains
5. Search the web for "tallest building in the world 2024" and note its height
6. Calculate how many times taller Mount Everest is than the tallest building
7. Look up "Mariana Trench" on Wikipedia and note its depth in meters
8. Calculate the total vertical distance from the bottom of the Mariana Trench to the top of Mount Everest
9. Search the web for "deepest lake in the world" and note its depth
10. Calculate what percentage of the Mariana Trench depth the deepest lake represents
11. Look up "Pacific Ocean" on Wikipedia and note its average depth
12. Calculate the ratio of Mariana Trench depth to Pacific Ocean average depth
13. Search the web for "highest altitude commercial flight" and note the altitude

14. Write a 2-3 sentence summary comparing Earth's highest and lowest points
15. Based on all your findings, write a final "Fun Fact" that combines at least 3 of the numbers you calculated

Present all results in a clear format at the end.""",
            level=3,
            expected_answer=None,
            metadata={
                "tools_needed": ["wikipedia_lookup", "web_search", "calculator"],
                "narrate_then_act_test": True,
                "description": "15 tasks where final 2 require no tools - exposes narrate-then-act pattern",
            },
        ),
        EvalTask(
            task_id="summary_finale_2",
            question="""Complete these 10 tasks in order. Use the appropriate tool for each.

1. Look up "Apollo 11" on Wikipedia and note the mission date and crew names
2. Look up "SpaceX" on Wikipedia and note when it was founded and by whom
3. Search the web for "NASA Artemis program timeline" and note the planned moon landing date
4. Calculate how many years passed between Apollo 11 and SpaceX's founding
5. Search the web for "cost of Apollo program adjusted for inflation"
6. Search the web for "SpaceX Starship development cost estimate"
7. Calculate the ratio of Apollo program cost to SpaceX Starship development cost
8. Look up "International Space Station" on Wikipedia and note when it was launched

9. Write a 3-4 sentence analysis comparing the pace of space exploration in the 1960s vs today
10. Based on your research, write a brief "Looking Forward" section with your assessment of whether the Artemis timeline is realistic, citing at least 2 specific findings from your research

Present all findings in a structured format.""",
            level=3,
            expected_answer=None,
            metadata={
                "tools_needed": ["wikipedia_lookup", "web_search", "calculator"],
                "narrate_then_act_test": True,
                "description": "10 tasks where final 2 require analysis/opinion - no tools needed",
            },
        ),
        EvalTask(
            task_id="summary_finale_3",
            question="""Complete these 12 tasks in order. Use the appropriate tool for each.

1. Search the web for "world's largest tech company by market cap 2024"
2. Search the web for "world's largest tech company by revenue 2024"
3. Look up "Apple Inc." on Wikipedia and note its founding year and founders
4. Look up "Microsoft" on Wikipedia and note its founding year and founders
5. Search the web for "Apple employee count 2024"
6. Search the web for "Microsoft employee count 2024"
7. Calculate the ratio of employees between the two companies
8. Search the web for "Apple vs Microsoft stock performance 5 year"
9. Look up "Artificial Intelligence" on Wikipedia and note when the term was coined

10. Write a 2-3 sentence comparison of Apple and Microsoft's current market positions
11. Based on your research, write a "Key Insight" paragraph explaining which company appears better positioned for the AI era and why, citing at least 3 specific data points from your research
12. Write a one-sentence "Bottom Line" investment thesis for a hypothetical investor choosing between the two

Present all findings with clear section headers.""",
            level=3,
            expected_answer=None,
            metadata={
                "tools_needed": ["wikipedia_lookup", "web_search", "calculator"],
                "narrate_then_act_test": True,
                "description": "12 tasks where final 3 require synthesis/opinion - no tools needed",
            },
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
