# Finish Condition Eval Harnesses

Evaluates different agent loop finish conditions across model providers.

## Structure

```
harness-evals/
├── harness_a.py    # Implicit finish (Claude Code style)
├── harness_b.py    # Explicit finish with todo validation (Alyx style)
├── harness_c.py    # Adaptive finish (hybrid)
├── tools.py        # Shared tool definitions
├── models.py       # Model provider abstraction (anthropic + openai)
├── run_eval.py     # Runs all combinations and collects results
└── README.md
```

## Harness Descriptions

### Harness A: Implicit Finish
**Finish condition:** Model stops calling tools.

The simplest approach — the model decides when it's done by simply not requesting any more tool calls.

### Harness B: Explicit Todo Validation
**Finish condition:** All tasks marked "completed" AND no more tool calls.

Forces the model to plan with todos and explicitly mark each task complete. Ensures systematic task completion.

### Harness C: Adaptive Finish
**Finish condition:** No tool calls AND (tasks completed OR done-language detected).

Hybrid approach that checks multiple signals: task completion status, absence of tool calls, and linguistic markers like "here's the result" or "I've completed".

## Setup

```bash
# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install anthropic openai python-dotenv pandas rich

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

## Usage

### Run all combinations
```bash
python run_eval.py
```

### Run specific harness/provider
```bash
python run_eval.py -H a -p anthropic
python run_eval.py -H b -p openai
```

### Custom prompts
```bash
python run_eval.py -P "What is 2+2?" -P "Find all errors in the trace"
```

### Save results
```bash
python run_eval.py -o results.json
```

### Run individual harness directly
```bash
python harness_a.py
python harness_b.py
python harness_c.py
```

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
```

## Results Format

Each eval run returns:
```python
{
    "text": str,              # Final response text
    "iterations": int,        # Number of loop iterations
    "tool_calls_total": int,  # Total tool calls made
    "finished_reason": str,   # Why the loop ended
    # Harness B/C also include:
    "tasks_created": int,
    "tasks_completed": int,
}
```
