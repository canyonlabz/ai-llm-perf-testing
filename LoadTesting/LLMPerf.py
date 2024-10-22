import argparse
from llmperf import LLMPerf

def main():
    parser = argparse.ArgumentParser(description="LLMPerf Benchmarking Script")
    parser.add_argument("--model", type=str, required=True, help="Model to benchmark")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--output_tokens", type=int, default=128, help="Number of output tokens")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    args = parser.parse_args()

    llm_perf = LLMPerf(
        model=args.model,
        prompt_file=args.prompt_file,
        iterations=args.iterations,
        output_tokens=args.output_tokens,
        concurrency=args.concurrency
    )

    results = llm_perf.run_benchmark()
    llm_perf.print_results(results)

if __name__ == "__main__":
    main()

    