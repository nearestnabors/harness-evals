# Findings: Finish Condition Harness Evaluation

This document summarizes what we learned from testing three different agent loop finish conditions across multiple models. Written for future agents to pick up this work.

## Test Setup

### Test 1: Sequential Calculator Operations
**Task:** 30 sequential calculator operations (start with 1000, perform 30 math operations, report each result). Models were explicitly instructed to use the calculator tool for every step.

### Test 2: Multi-Step Research with Summary Finale
**Task:** 15 tasks mixing Wikipedia lookups, web searches, and calculations, where the **final 2 tasks require no tools** (writing a summary and a "fun fact"). This task structure is critical for exposing the narrate-then-act failure mode.

**Models tested:**
- GPT-4o (OpenAI)
- Claude Sonnet 4 (Anthropic)
- Gemma 4 31B (Google, via OpenRouter)

**Harnesses tested:**
- **Implicit:** No tool calls = done (simplest)
- **Explicit:** Must call `finish()` tool to end
- **Adaptive:** Detects "narrate-then-act" patterns, sends escalating nudges

## Results

### Test 1: Sequential Calculator (30 operations)

| Model | Harness | Tool Calls | Tokens | Completed |
|-------|---------|------------|--------|-----------|
| GPT-4o | Implicit | 30/30 | 58K | Yes |
| GPT-4o | Explicit | 31 | 52K | Yes |
| GPT-4o | Adaptive | 30/30 | 59K | Yes |
| Claude Sonnet 4 | Implicit | 30/30 | 82K | Yes |
| Claude Sonnet 4 | Explicit | 32 | 94K | Yes |
| Claude Sonnet 4 | Adaptive | 30/30 | 82K | Yes |

### Test 2: Multi-Step Research + Summary (15 tasks)

#### Before: Original Adaptive Harness

| Model | Harness | Iterations | Tool Calls | False Finishes | Task 15 Done | Exit Reason |
|-------|---------|------------|------------|----------------|--------------|-------------|
| Claude Sonnet 4 | Implicit | 18 | 17 | 0 | **Yes** | no_tool_calls |
| GPT-4o | Implicit | 13 | 14 | 0 | **No** | no_tool_calls |
| Claude Sonnet 4 | Explicit | 15 | 15 | 0 | **Yes** | finish_validated |
| GPT-4o | Explicit | 14 | 15 | 0 | **No** | finish_validated |
| Claude Sonnet 4 | Adaptive | 20 | 18 | 2 (false positive) | **Yes** | max_iterations |
| GPT-4o | Adaptive | 13 | 13 | 1 (true positive) | **Yes** | genuine_completion |

#### After: Improved Adaptive Harness (position-aware detection)

| Model | Harness | Iterations | Tool Calls | False Finishes | Task 15 Done | Exit Reason |
|-------|---------|------------|------------|----------------|--------------|-------------|
| Claude Sonnet 4 | Adaptive | 15 | 14 | **0** | **Yes** | genuine_completion |
| GPT-4o | Adaptive | 14 | 14 | 1 (true positive) | **Yes** | genuine_completion |

The improved adaptive harness eliminates Claude's false positives while still catching GPT-4o's narrate-then-act pattern.

## Key Findings

### 1. Task structure determines whether false finishes occur

The calculator task (Test 1) didn't expose the narrate-then-act pattern because every step required a tool call. The research+summary task (Test 2) exposed it because the final tasks required **synthesis without tools**.

**GPT-4o's failure in Test 2 (Implicit):**
```
Finally, I'll write a "Fun Fact" that combines at least three of the numbers calculated.

✅ EXIT: No tool calls in response
```

GPT-4o announced it would write the fun fact, then stopped without writing it. This is the classic narrate-then-act pattern.

### 2. The explicit harness catches exits but doesn't force completion

With the explicit harness, GPT-4o got nudged after the premature exit attempt. But it simply called `finish()` without completing task 15. The explicit harness prevents silent failures but doesn't guarantee task completion.

### 3. The adaptive harness works for GPT-4o but has false positives for Claude

**GPT-4o with Adaptive:** The harness detected `"I'll write"` as narrate-then-act and nudged:
```
🔍 FALSE-FINISH DETECTED: ["I'll write", "I'll write"]
💬 NUDGE (1/3): Continue with your plan.
```
GPT-4o then completed the fun fact. **Success.**

**Claude with Adaptive:** Claude completed all tasks but got flagged for phrases like `"Let me provide"` at the end of its response—even though it had already written the content. This caused unnecessary nudges and hit max iterations. **False positive.**

The difference: **Claude's narration accompanies action; GPT-4o's narration replaces action.**

### 4. Position-aware detection fixes the adaptive harness

The original detection flagged any occurrence of "Let me..." or "I'll..." anywhere in the response. The fix: only flag patterns that appear **at the end** of a response with **no substantive content following**. This distinguishes between:

- **Claude:** "Let me write a summary. [500 words of actual summary]" → Pattern early, content follows → **Not flagged**
- **GPT-4o:** "Finally, I'll write a fun fact." → Pattern at end, nothing follows → **Flagged**

## Harness Recommendations (Early Findings)

| Scenario | Recommended Harness | Reasoning |
|----------|---------------------|-----------|
| Claude-only deployment | Implicit | Simple, cheap, reliable |
| GPT-only deployment | Adaptive | Catches narrate-then-act |
| Gemma 4 deployment | Implicit or Adaptive | Both work well with native function calling |
| Multi-model deployment | Adaptive | Works for all with position-aware detection |
| Debugging any model | Adaptive | Reveals behavior patterns in traces |

*See Test 4 results below for updated recommendations based on 117-run benchmark.*

## Implemented: Position-Aware Adaptive Harness

The original adaptive harness had false positives because it didn't consider **where** the narration appears. The fix (now implemented in `harness_adaptive.py`):

1. **Position-aware detection:** Only check the last ~300 characters of a response
2. **Content-after-pattern check:** Only flag if < 100 characters follow the pattern
3. **Completion signals still respected:** Phrases like "Final Answer:" bypass detection entirely

```python
# From harness_adaptive.py - _detect_narrate_then_act()
def _detect_narrate_then_act(text: str) -> tuple[bool, list[str]]:
    if _has_completion_signal(text):
        return False, []

    # Only check the TRAILING portion of the response
    trailing_window = 300   # chars from end to check
    min_content_after = 100 # chars after pattern to be "safe"

    trailing_text = text[-trailing_window:] if len(text) > trailing_window else text

    matched = []
    for pattern in NARRATE_PATTERNS_COMPILED:
        for match in pattern.finditer(trailing_text):
            # Only flag if pattern is near the very end
            chars_after = len(trailing_text) - match.end()
            if chars_after < min_content_after:
                matched.append(match.group(0))

    return len(matched) > 0, matched[:3]
```

**Results after fix:**
| Model | False Finishes (Before) | False Finishes (After) |
|-------|------------------------|------------------------|
| Claude | 2 (false positives) | **0** |
| GPT-4o | 1 (true positive) | 1 (true positive) |

## Open Questions

1. **Should we combine adaptive + explicit?** Require `finish()` but only after adaptive checks pass. (Note: explicit harness adds overhead and can cause failures, so this may not be worthwhile.)

2. **What about streaming responses?** The narrate-then-act pattern might be detectable mid-stream before the response completes.

3. **Model version matters:** Gemma 3 couldn't make structured tool calls; Gemma 4 can. Always verify tool calling works with your specific model version.

## Test 3: Full Benchmark Suite (April 2026)

Ran 5 representative tasks across all harnesses with frontier models.

### Benchmark Configuration
- **Tasks:** lookup_calc (3), research_synthesis (2)
- **Harnesses:** implicit, explicit, adaptive
- **Providers:** anthropic (Claude Sonnet 4), openai (GPT-4o)
- **Total runs:** 30

### Results by Harness

| Harness | Completed | Correct | False Finishes | Total Tokens |
|---------|-----------|---------|----------------|--------------|
| Implicit | 10/10 | 2/6 | 0 | 109K |
| Explicit | 10/10 | 2/6 | 0 | 135K |
| Adaptive | 10/10 | 2/6 | 0 | 106K |

### Results by Provider

| Provider | Completed | Correct | False Finishes | Total Tokens |
|----------|-----------|---------|----------------|--------------|
| Claude Sonnet 4 | 15/15 | 3/9 | 0 | 253K |
| GPT-4o | 15/15 | 3/9 | 0 | 97K |

### Key Observation
The benchmark tasks (lookup+calculate, research) all ended with tool calls, so they didn't expose the narrate-then-act pattern. Only the summary-finale tasks reveal it.

### Targeted Summary-Finale Test

Ran the summary_finale_1 task specifically to test narrate-then-act:

| Model | Harness | Task 15 Done | Exit Reason |
|-------|---------|--------------|-------------|
| Claude | Implicit | **Yes** | no_tool_calls |
| GPT-4o | Implicit | **No** | no_tool_calls |
| Claude | Explicit | **Yes** | finish_validated |
| GPT-4o | Explicit | **Yes** (in finish) | finish_validated |
| Claude | Adaptive | **Yes** | genuine_completion |
| GPT-4o | Adaptive | **Yes** | genuine_completion |

**GPT-4o's intermittent failure with implicit harness:**
```
Finally, I'll write a "Fun Fact" that combines at least three of the numbers we calculated. Let's proceed with that.

✅ EXIT: No tool calls in response
```

GPT-4o announced intent, then stopped. The adaptive harness catches this when it occurs.

## Eval Task Suite

The evaluation suite (`eval_tasks.py`) contains 13 tasks across 6 categories:

| Category | Count | Task IDs | Tools | Narrate-Then-Act Test |
|----------|-------|----------|-------|----------------------|
| Lookup + Calculate | 3 | `lookup_calc_1-3` | web_search, calculator | No |
| Research + Synthesis | 2 | `research_synthesis_1-2` | web_search, wikipedia | No |
| File Analysis | 2 | `file_analysis_1-2` | read_file, calculator | No |
| Iterative Refinement | 2 | `iterative_1-2` | web_search | No |
| Comparison | 1 | `comparison_1` | web_search | No |
| **Summary Finale** | **3** | `summary_finale_1-3` | all tools | **Yes** |

### Summary-Finale Tasks (Narrate-Then-Act Tests)

These tasks are specifically designed to expose the narrate-then-act pattern:

1. **summary_finale_1** (10 tasks): Earth geography research ending with summary + fun fact
2. **summary_finale_2** (10 tasks): Space exploration research ending with analysis + assessment
3. **summary_finale_3** (12 tasks): Tech company comparison ending with analysis + insight + investment thesis

Each task has multiple tool-requiring steps followed by 2-3 synthesis steps that require **no tools**. This structure exposes models that announce "I'll write..." and then stop without writing.

## Code Pointers

- `harness_implicit.py` - Simplest loop, exits on no tool calls
- `harness_explicit.py` - Requires `finish()` tool, nudges if text-only response
- `harness_adaptive.py` - Narrate-then-act detection with escalating nudges (position-aware)
- `models.py` - Provider abstraction (Anthropic, OpenAI, OpenRouter)
- `tools.py` - Calculator, web search, Wikipedia, file read
- `eval_tasks.py` - 13 evaluation tasks including 3 summary-finale narrate-then-act tests
- `run_benchmark.py` - Full benchmark runner with JSON output

## Running the Tests

```bash
# Single harness, single model
python harness_adaptive.py --provider openai --model gpt-4o -P "your prompt"

# Full benchmark (frontier models)
python run_benchmark.py --provider frontier --output results.json

# Full benchmark (all models including OpenRouter)
python run_benchmark.py --provider all --output results.json

# Specific task count
python run_benchmark.py --provider frontier --max-tasks 5

# The summary-finale test (exposes narrate-then-act)
python harness_implicit.py --provider openai -P "Complete these 15 tasks in order..."
```

## Test 4: Full Benchmark (117 runs)

Ran the complete 13-task suite across all three harnesses with all three providers.

### Configuration
- **Tasks:** All 13 (lookup_calc, research_synthesis, file_analysis, iterative, comparison, summary_finale)
- **Harnesses:** implicit, explicit, adaptive
- **Providers:** Anthropic (Claude Sonnet 4), OpenAI (GPT-4o), OpenRouter (Gemma 4 31B)
- **Total runs:** 117 (13 tasks × 3 providers × 3 harnesses)

### Results by Harness

| Harness | Completed | Correct | Avg Iters | Avg Tools | False Finishes | Total Tokens |
|---------|-----------|---------|-----------|-----------|----------------|--------------|
| Implicit | 39/39 (100%) | 3/9 | 5.5 | 4.9 | - | 858K |
| Adaptive | 39/39 (100%) | **4/9** | 5.4 | 4.8 | **2** | 913K |
| Explicit | 39/39 (100%) | 4/9 | 5.8 | 5.9 | 0 | 965K |

### Key Finding: Adaptive Harness Catches Real False Finishes

The adaptive harness detected **2 false finishes** on OpenAI (GPT-4o):
- `comparison_1`: GPT-4o said "I'll create a comparison table" then stopped
- `summary_finale_3`: GPT-4o announced intent to write analysis then stopped

Both times, the adaptive harness nudged GPT-4o to continue, and the task completed successfully. **The implicit harness would have exited prematurely.**

### Correctness Improvement

The adaptive harness achieved **4/9 correct answers** vs implicit's **3/9**. The extra correct answer came from a task where adaptive's nudge allowed GPT-4o to complete work that implicit would have missed.

### Token Efficiency

| Harness | Total Tokens | Avg per Task |
|---------|--------------|--------------|
| Implicit | 858K | 22.0K |
| Adaptive | 913K | 23.4K |
| Explicit | 965K | 24.7K |

Adaptive costs ~6% more than implicit but catches false finishes. Explicit costs ~12% more with no additional benefit.

### All Three Models Work

| Model | Tool Calling | Completion Rate | Best Harness |
|-------|--------------|-----------------|--------------|
| Claude Sonnet 4 | Native ✓ | 100% | Any (no false finishes observed) |
| GPT-4o | Native ✓ | 100% | **Adaptive** (catches narrate-then-act) |
| Gemma 4 31B | Native ✓ | 100% | Any (no false finishes observed) |

## Harness Comparison Summary

| Harness | Completion | Correctness | Cost | False Finish Detection |
|---------|------------|-------------|------|------------------------|
| Implicit | 100% | 3/9 | Cheapest | ❌ None |
| **Adaptive** | 100% | **4/9** | +6% | ✅ Catches GPT-4o issues |
| Explicit | 100% | 4/9 | +12% | ❌ None (just adds overhead) |

**Key insight:** The adaptive harness is the best choice for multi-model deployments. It catches real false finishes (proven with GPT-4o), improves correctness, and costs only marginally more than implicit. The explicit harness adds cost without catching any issues the adaptive harness doesn't already catch.

## Summary

**The narrate-then-act problem is real and the adaptive harness catches it.** In our 117-run benchmark, the adaptive harness caught 2 false finishes from GPT-4o that would have caused task failures with the implicit harness. This translated to improved correctness (4/9 vs 3/9).

**The position-aware adaptive harness works without false positives.** Earlier versions flagged Claude's "Let me..." phrases even when Claude had already completed the work. The position-aware fix (only flag patterns at the end with no content following) eliminates these false positives.

**All three models now work reliably.** Claude Sonnet 4, GPT-4o, and Gemma 4 31B all achieved 100% completion across all harnesses.

**Recommendations for production agent loops:**

| Scenario | Recommended | Reasoning |
|----------|-------------|-----------|
| GPT-4o deployment | **Adaptive** | Proven to catch narrate-then-act (2 instances in benchmark) |
| Claude deployment | Implicit or Adaptive | No false finishes observed, adaptive adds minimal overhead |
| Gemma 4 deployment | Implicit or Adaptive | No false finishes observed |
| Multi-model deployment | **Adaptive** | Best coverage across all models |
| Debugging any model | **Adaptive** | Reveals behavior patterns in Phoenix traces |

**The explicit harness is not recommended.** It costs 12% more than implicit and doesn't catch any issues that adaptive doesn't already catch. It just adds overhead.
