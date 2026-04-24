# Finish Condition Eval Harnesses

Evaluates different agent loop finish conditions across model providers.

## Structure

```
harness-evals/
├── harness_a.py      # Implicit finish (Claude Code style)
├── harness_b.py      # Explicit finish with finish() tool
├── harness_c.py      # Adaptive finish (hybrid with smart detection)
├── tools.py          # Mock trace/span tools (default)
├── tools_gaia.py     # Real tools: web search, calculator, Wikipedia
├── gaia_loader.py    # GAIA benchmark task loader
├── models.py         # Model provider abstraction (anthropic + openai)
├── run_eval.py       # Run eval matrix with mock tools
├── run_gaia.py       # Run GAIA benchmark with real tools
├── instrumentation.py # Phoenix tracing setup
└── README.md
```

## Harness Descriptions

### Harness A: Implicit Finish
**Finish condition:** Model stops calling tools.

The simplest approach — the model decides when it's done by simply not requesting any more tool calls. Cheap and effective, but can't catch premature exits.

### Harness B: Explicit Finish
**Finish condition:** Model must call `finish()` tool.

Forces the model to explicitly signal completion. Adds ~5% token overhead but provides clear intent.

### Harness C: Adaptive Finish
**Finish condition:** Smart detection of genuine completion vs. false finishes.

Hybrid approach that:
1. Detects "narrate-then-act" patterns ("I'll search for..." without calling tools)
2. Recognizes completion signals ("Final Answer:", checkmarks, etc.)
3. Uses escalating nudges when stuck
4. Falls back to force-exit after 3 consecutive false finishes

## Results Summary

Tested on GAIA-style benchmark tasks (multi-step reasoning with web search, Wikipedia, calculator):

| Harness | Completion | Correct | Avg Iters | Tokens | Notes |
|---------|------------|---------|-----------|--------|-------|
| **A** (Implicit) | 100% | 2/5 | 5.8 | 67,065 | Simple, no safety net |
| **B** (Explicit) | 100% | 2/5 | 6.0 | 70,278 | +5% overhead from finish() |
| **C** (Adaptive) | 100% | 2/5 | 5.8 | 64,932 | Smart detection, slight efficiency gain |

**Key finding:** Harness C matches or beats A's efficiency while providing safety guarantees against false finishes.

## Setup

```bash
# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install anthropic openai python-dotenv rich \
    arize-phoenix openinference-instrumentation-anthropic openinference-instrumentation-openai \
    tavily-python datasets  # For GAIA benchmark

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY - required
#   OPENAI_API_KEY - required for OpenAI models
#   TAVILY_API_KEY - required for web search (get free at tavily.com)
#   HF_TOKEN - optional, for real GAIA tasks from HuggingFace
```

## Tracing

All harness runs are automatically traced to a **local Phoenix server** at http://localhost:6006.

Traces include:
- **CHAIN spans** for each agent run and iteration
- **LLM spans** (auto-instrumented) for each API call
- **TOOL spans** for each tool invocation with input/output

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
| `todos_abandoned` | int | Incomplete todos at exit (B/C only) |

### Viewing Traces

Start Phoenix in a separate terminal first:

```bash
source venv/bin/activate
python -m phoenix.server.main serve
```

Then run harnesses in another terminal. Open http://localhost:6006 to see your traces.

## Usage

### GAIA Benchmark (recommended)

Run with sample tasks (no HuggingFace access needed):
```bash
# All harnesses
python run_gaia.py --sample

# Single harness
python run_gaia.py --sample --harness C

# With OpenAI
python run_gaia.py --sample --provider openai
```

Run with real GAIA tasks (requires HF_TOKEN):
```bash
python run_gaia.py --level 1 --max-tasks 5
```

### Mock Tools (original eval)

```bash
# All combinations
python run_eval.py

# Specific harness/provider
python run_eval.py -H a -p anthropic

# Multiple trials
python run_eval.py --trials 3
```

### Run individual harness directly
```bash
python harness_a.py
python harness_b.py
python harness_c.py
```

## GAIA Tools

Real tools for benchmark tasks:

| Tool | Description | Requires |
|------|-------------|----------|
| `web_search` | Search the web via Tavily | `TAVILY_API_KEY` |
| `wikipedia_lookup` | Get Wikipedia article summaries | - |
| `calculator` | Safe math evaluation | - |
| `read_file` | Read file contents | - |

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
    "iterations": int,        # Number of loop iterations
    "tool_calls_total": int,  # Total tool calls made
    "total_tokens": int,      # Total tokens used
    "false_finishes": int,    # False finishes detected (C only)
    "narrate_then_act": int,  # Narrate-then-act events (C only)
    "task_complete": bool,    # Did model complete the task?
    "finished_reason": str,   # Why the loop ended
}
```

## Adaptive Finish Details (Harness C)

Harness C uses two detection mechanisms:

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
- Checkmarks (✅)
- "The answer is..."
- "In conclusion..."

When completion signals are present, narrate-then-act patterns are ignored (prevents false positives).
