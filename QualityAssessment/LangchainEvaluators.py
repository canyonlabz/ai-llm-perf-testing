import argparse
"""
Langchain Evaluator Script
This script evaluates prompts using a specified OpenAI model and a Langchain evaluator. 
It supports concurrent evaluations and multiple iterations.
Command-line Arguments:
    --model (str): OpenAI model to use (default: "text-davinci-002").
    --concurrency (int): Number of concurrent evaluations (default: 1).
    --iterations (int): Number of iterations (default: 10).
Example Usage:
    python LangchainEvaluators.py --model text-davinci-003 --concurrency 5 --iterations 20
"""
from langchain.evaluation import load_evaluator
from langchain.llms import OpenAI
from concurrent.futures import ThreadPoolExecutor
import time

#
# evaluate_prompt(evaluator, llm, prompt, reference):
#        Input: Evaluates a single prompt using the provided evaluator and language model.
#        Output: Returns a dictionary containing the prompt, prediction, evaluation result, and response time.
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

#
# main():
#        - Parses command-line arguments to configure the model, concurrency, and iterations.
#        - Loads the specified OpenAI model and Langchain evaluator.
#        - Evaluates a set of sample prompts concurrently and prints the average response time and total evaluations.
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
    # Evaluate prompts concurrently using a ThreadPoolExecutor with the specified concurrency
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        # Submit tasks for each prompt and reference pair for the specified number of iterations
        for _ in range(args.iterations):
            for prompt, reference in zip(prompts, references): # Evaluate each prompt-reference pair concurre
                future = executor.submit(evaluate_prompt, evaluator, llm, prompt, reference) # Evaluate the prompt using the evaluator and language model
                futures.append(future) # Append the future to the list of futures
        
        for future in futures:
            results.append(future.result()) # Append the result of the future to the list of results

    # Analyze results
    total_time = sum(r["response_time"] for r in results)
    avg_time = total_time / len(results)
    print(f"Average response time: {avg_time:.2f} seconds")
    print(f"Total evaluations: {len(results)}")

    # You can add more detailed analysis here

if __name__ == "__main__":
    main()