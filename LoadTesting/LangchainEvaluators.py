import argparse
from langchain.evaluation import load_evaluator
from langchain.llms import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time

def evaluate_prompt(evaluator, llm, prompt, reference):
    start_time = time.time()
    prediction = llm(prompt)
    eval_result = evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference,
    )
    end_time = time.time()
    return {
        "prompt": prompt,
        "prediction": prediction,
        "evaluation": eval_result,
        "response_time": end_time - start_time
    }

def main():
    parser = argparse.ArgumentParser(description="Langchain Evaluator Script")
    parser.add_argument("--model", type=str, default="text-davinci-002", help="OpenAI model to use")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent evaluations")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    llm = OpenAI(model_name=args.model)
    evaluator = load_evaluator("qa")

    # Sample prompts and references (replace with your own dataset)
    prompts = [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        # Add more prompts here
    ]
    references = [
        "The capital of France is Paris.",
        "William Shakespeare wrote 'Romeo and Juliet'.",
        # Add corresponding references here
    ]

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for _ in range(args.iterations):
            for prompt, reference in zip(prompts, references):
                future = executor.submit(evaluate_prompt, evaluator, llm, prompt, reference)
                futures.append(future)
        
        for future in futures:
            results.append(future.result())

    # Analyze results
    total_time = sum(r["response_time"] for r in results)
    avg_time = total_time / len(results)
    print(f"Average response time: {avg_time:.2f} seconds")
    print(f"Total evaluations: {len(results)}")

    # You can add more detailed analysis here

if __name__ == "__main__":
    main()