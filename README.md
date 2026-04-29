# Finish Condition Eval Harnesses

A benchmark suite for evaluating how different agent loop finish conditions affect task completion across LLM providers.

## Why This Exists

When building agentic applications, you need to decide how the agent loop knows when to stop. The simplest approach—"stop when the model doesn't call any tools"—has a subtle failure mode: **the model says it will do something, then stops without doing it.**

```
User: Find Tokyo's population and multiply by 2.

Model: "I'll search for Tokyo's population now."
       [NO TOOL CALL - just stops here]
```

This is a **false finish**—the model signals completion but hasn't actually completed the task. This suite lets you test different finish conditions to see which ones catch this pattern for your models.

## Built With

| Tool | Purpose |
|------|---------|
| [Phoenix](https://github.com/Arize-ai/phoenix) | Tracing and eval analysis—every harness run is traced with custom attributes for debugging |
| [Tavily](https://tavily.com) | Web search tool for agents |
| [OpenInference](https://github.com/Arize-ai/openinference) | Auto-instrumentation for Anthropic and OpenAI calls |

All harness runs are automatically traced to a local Phoenix server, letting you inspect each iteration, tool call, and false finish detection in detail.

## Structure

```
harness-evals/
├── harness_implicit.py   # Implicit finish (no tool calls = done)
├── harness_explicit.py   # Explicit finish (requires finish() tool)
├── harness_adaptive.py   # Adaptive finish (detects false finishes)
├── tools.py              # Web search, Wikipedia, calculator
├── eval_tasks.py         # Multi-step evaluation tasks
├── models.py             # Model provider abstraction (Anthropic + OpenAI)
├── evals.py              # Quick single-prompt evaluation
├── run_benchmark.py      # Full benchmark suite
├── instrumentation.py    # Phoenix tracing setup
└── README.md
```

## Harness Descriptions

### Implicit Finish (`harness_implicit.py`)
**Finish condition:** Model stops calling tools.

The simplest approach — the model decides when it's done by simply not requesting any more tool calls. Cheap and effective, but can't catch premature exits (false finishes).

### Explicit Finish (`harness_explicit.py`)
**Finish condition:** Model must call `finish()` tool.

Adds a `finish()` tool that the model must call to signal completion:

```
Model: "Here's the answer: 28 million."
       [calls finish(summary="Calculated Tokyo population * 2")]
```

**Pros:**
- Clear intent — no ambiguity about whether the model thinks it's finished
- Can include a summary of what was accomplished

**Cons:**
- ~5% token overhead (extra tool call)
- Model might forget to call it (then you need a nudge anyway)
- Doesn't solve the core problem — model can still call `finish()` prematurely

### Adaptive Finish (`harness_adaptive.py`)
**Finish condition:** Smart detection of genuine completion vs. false finishes.

Hybrid approach that:
1. Detects "narrate-then-act" patterns ("I'll search for..." without calling tools)
2. Recognizes completion signals ("Final Answer:", checkmarks, etc.)
3. Uses escalating nudges when stuck
4. Falls back to force-exit after 3 consecutive false finishes

## Setup

```bash
# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install anthropic openai python-dotenv rich \
    arize-phoenix openinference-instrumentation-anthropic openinference-instrumentation-openai \
    tavily-python wikipedia

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY - required
#   OPENAI_API_KEY - required for OpenAI models
#   TAVILY_API_KEY - required for web search (get free at tavily.com)
```

## Usage

### Quick Test (single prompt)

```bash
# All harnesses with Anthropic
python evals.py

# Both providers
python evals.py --provider both

# Specific harness with verbose output
python evals.py --harness adaptive -v

# Custom prompt
python evals.py -P "What is 2+2? Use the calculator."
```

### Full Benchmark

```bash
# All harnesses, Anthropic only
python run_benchmark.py

# Both providers (full matrix)
python run_benchmark.py --provider both

# Specific harness
python run_benchmark.py --harness adaptive

# Limit tasks
python run_benchmark.py --max-tasks 3

# Quick narrate-then-act test (summary_finale tasks only)
# These tasks end with synthesis steps that need NO tools,
# exposing models that say "I'll write..." then stop without writing.
python run_benchmark.py --narrate-test

# Filter by task category
python run_benchmark.py --category lookup_calc
```

### Quick Narrate-Then-Act Test

To quickly test whether a model exhibits the narrate-then-act pattern:

```bash
# Run summary_finale tasks with all harnesses
python run_benchmark.py --narrate-test --provider anthropic
python run_benchmark.py --narrate-test --provider openai
python run_benchmark.py --narrate-test --provider openrouter  # Gemma 4

# Compare implicit vs adaptive to see if adaptive catches false finishes
python run_benchmark.py --narrate-test --harness implicit --provider openai
python run_benchmark.py --narrate-test --harness adaptive --provider openai
```

The `--narrate-test` flag runs only the `summary_finale` tasks, which are specifically designed to expose models that announce intent ("I'll write a summary...") without following through.

### Supported Providers

| Provider | Model | Flag |
|----------|-------|------|
| Anthropic | Claude Sonnet 4 | `--provider anthropic` |
| OpenAI | GPT-4o | `--provider openai` |
| OpenRouter | Gemma 4 31B | `--provider openrouter` |

Run all providers at once:
```bash
# All three providers
python run_benchmark.py --provider all

# Just frontier models (Anthropic + OpenAI)
python run_benchmark.py --provider frontier
```

### Run Individual Harness

```bash
python harness_implicit.py
python harness_explicit.py
python harness_adaptive.py
```

## Tracing

All harness runs are automatically traced to a **local Phoenix server** at http://localhost:6006.

### Start Phoenix

```bash
source venv/bin/activate
python -m phoenix.server.main serve
```

Then run harnesses in another terminal. Open http://localhost:6006 to see your traces.

### Custom Attributes

Each span includes custom attributes for eval analysis:

| Attribute | Type | Description |
|-----------|------|-------------|
| `harness_type` | string | "implicit" \| "explicit" \| "adaptive" |
| `model_provider` | string | "anthropic" \| "openai" |
| `model_name` | string | Specific model used |
| `iteration_number` | int | Which loop iteration (1-indexed) |
| `finish_triggered` | bool | Did this iteration trigger a finish? |
| `false_finish` | bool | Was a false finish detected? |
| `narrate_then_act` | bool | Did model narrate without acting? |
| `task_complete` | bool | Was task actually done at exit? |

## Tools

| Tool | Description | Requires |
|------|-------------|----------|
| `web_search` | Search the web via Tavily | `TAVILY_API_KEY` |
| `wikipedia_lookup` | Get Wikipedia article summaries | - |
| `calculator` | Safe math evaluation | - |

## Model Abstraction

The `models.py` module provides a unified interface:

```python
from models import call_model, ModelResponse

response: ModelResponse = call_model(
    provider="anthropic",  # or "openai"
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    tools=tools_list,
    system_prompt="You are helpful.",
)

print(response.text)        # str
print(response.tool_calls)  # list[ToolCall]
print(response.tokens)      # TokenUsage (input_tokens, output_tokens)
```

## Results Format

Each eval run returns:
```python
{
    "text": str,              # Final response text
    "iterations": list,       # Iteration details for tracing
    "iteration_count": int,   # Number of loop iterations
    "tool_calls_total": int,  # Total tool calls made
    "total_tokens": int,      # Total tokens used
    "false_finishes": int,    # False finishes detected (adaptive only)
    "narrate_then_act": int,  # Narrate-then-act events (adaptive only)
    "task_complete": bool,    # Did model complete the task?
    "finished_reason": str,   # Why the loop ended
}
```

## Adaptive Finish Details

The adaptive harness uses two detection mechanisms:

### 1. Narrate-then-act detection
Catches phrases like:
- "I'll search for..."
- "Let me check..."
- "Now I will..."

When detected without tool calls, sends escalating nudges:
1. "Continue with your plan."
2. "You said '[phrase]' but didn't call any tools. Please execute the action now."
3. "IMPORTANT: Call a tool or provide your final answer."

### 2. Completion signal detection
Recognizes legitimate completion:
- "Final Answer:"
- Checkmarks (checkmark emoji)
- "The answer is..."
- "In conclusion..."

When completion signals are present, narrate-then-act patterns are ignored (prevents false positives).

## Future Enhancement: Task Management Integration

The adaptive harness becomes even more effective when combined with a **task management tool**. Instead of relying on pattern matching to detect false finishes, you can verify actual task completion state.

### How It Would Work

1. Add a `todo` tool that lets the model create and complete tasks
2. When the model stops calling tools, check if any tasks are still incomplete
3. If incomplete tasks exist, nudge with specifics: "Task 3 (Calculate average) is still pending"
4. Fall back to pattern matching if the model doesn't use the task tool

### Benefits

- **Concrete verification** instead of pattern guessing
- **Specific nudges** ("Task 3 still pending") vs generic ("Continue with your plan")
- **Graceful fallback** to current behavior if model doesn't use task tool
- **Works with any model** that supports tool calling

This is particularly effective for multi-step tasks where the prompt contains a numbered list of steps. The harness could even pre-populate tasks from the prompt and require the model to mark each one complete.
