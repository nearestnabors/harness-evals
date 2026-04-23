"""
Eval runner: runs all harness × provider combinations and collects results.
"""

import argparse
import json
from datetime import datetime
from typing import Any

import harness_a
import harness_b
import harness_c
from models import DEFAULT_MODELS, Provider

# Available harnesses
HARNESSES = {
    "a": ("implicit", harness_a),
    "b": ("explicit_todo", harness_b),
    "c": ("adaptive", harness_c),
}

# Available providers
PROVIDERS: list[Provider] = ["anthropic", "openai"]

# Default test prompts
DEFAULT_PROMPTS = [
    "What is 2 + 2?",
    "Find all spans containing 'Aragorn' and list their IDs.",
    "Get a preview of the trace and tell me how many spans there are.",
    "Find spans with errors and calculate the total latency.",
]


def run_single(
    harness_name: str,
    provider: Provider,
    prompt: str,
    model: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single harness with a single provider and prompt."""
    _, harness_module = HARNESSES[harness_name]

    result = harness_module.run(
        prompt=prompt,
        provider=provider,
        model=model,
        verbose=verbose,
    )

    return {
        "harness": harness_name,
        "provider": provider,
        "model": model or DEFAULT_MODELS[provider],
        "prompt": prompt,
        **result,
    }


def run_eval(
    prompts: list[str] | None = None,
    harnesses: list[str] | None = None,
    providers: list[Provider] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Run evaluation across all combinations.

    Args:
        prompts: List of prompts to test (defaults to DEFAULT_PROMPTS)
        harnesses: List of harness names to test (defaults to all)
        providers: List of providers to test (defaults to all)
        verbose: Print detailed output

    Returns:
        List of result dicts
    """
    prompts = prompts or DEFAULT_PROMPTS
    harnesses = harnesses or list(HARNESSES.keys())
    providers = providers or PROVIDERS

    results = []

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:50]}...")
        print("=" * 60)

        for harness_name in harnesses:
            harness_label, _ = HARNESSES[harness_name]

            for provider in providers:
                print(f"\n  [{harness_label}] [{provider}]")

                try:
                    result = run_single(
                        harness_name=harness_name,
                        provider=provider,
                        prompt=prompt,
                        verbose=verbose,
                    )
                    results.append(result)

                    print(f"    Iterations: {result['iterations']}")
                    print(f"    Tool calls: {result['tool_calls_total']}")
                    print(f"    Finished: {result['finished_reason']}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append(
                        {
                            "harness": harness_name,
                            "provider": provider,
                            "prompt": prompt,
                            "error": str(e),
                        }
                    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run finish-condition evals")
    parser.add_argument(
        "--harness",
        "-H",
        choices=list(HARNESSES.keys()),
        action="append",
        help="Harness(es) to run (default: all)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=PROVIDERS,
        action="append",
        help="Provider(s) to use (default: all)",
    )
    parser.add_argument(
        "--prompt",
        "-P",
        action="append",
        help="Custom prompt(s) to test",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    results = run_eval(
        prompts=args.prompt,
        harnesses=args.harness,
        providers=args.provider,
        verbose=args.verbose,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for harness_name in args.harness or list(HARNESSES.keys()):
        harness_label, _ = HARNESSES[harness_name]
        harness_results = [r for r in results if r.get("harness") == harness_name]

        if not harness_results:
            continue

        print(f"\n{harness_label}:")
        for provider in args.provider or PROVIDERS:
            provider_results = [r for r in harness_results if r.get("provider") == provider]
            if not provider_results:
                continue

            avg_iterations = sum(r.get("iterations", 0) for r in provider_results) / len(
                provider_results
            )
            avg_tools = sum(r.get("tool_calls_total", 0) for r in provider_results) / len(
                provider_results
            )
            errors = sum(1 for r in provider_results if "error" in r)

            print(f"  {provider}: avg_iter={avg_iterations:.1f}, avg_tools={avg_tools:.1f}, errors={errors}")

    # Save results
    if args.output:
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
