import time
from prompttools.experiment import Experiment
from prompttools.mock import MockLLM
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Replace MockLLM with your actual LLM integration
# For example: from langchain.llms import OpenAI

def generate_prompts(complexity, num_prompts):
    if complexity == "simple":
        return [f"What is {i}+{i}?" for i in range(num_prompts)]
    elif complexity == "medium":
        return [f"Explain the concept of {['gravity', 'photosynthesis', 'democracy'][i%3]} in one sentence." for i in range(num_prompts)]
    else:  # complex
        return [f"Compare and contrast the economic policies of {['USA', 'China', 'Germany'][i%3]} and {['Japan', 'Brazil', 'India'][i%3]}." for i in range(num_prompts)]

def run_experiment(complexity, num_prompts, concurrency):
    prompts = generate_prompts(complexity, num_prompts)
    llm = MockLLM()  # Replace with your actual LLM

    experiment = Experiment(
        prompts=prompts,
        llms=[llm],
        n=1  # Number of runs per prompt
    )

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(lambda p: experiment.run_prompt(p, llm), prompts))
    end_time = time.time()

    results = experiment.get_results()
    total_time = end_time - start_time
    avg_time = total_time / len(prompts)

    return {
        "complexity": complexity,
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "total_time": total_time,
        "avg_time": avg_time,
        "results": results
    }

def main():
    complexities = ["simple", "medium", "complex"]
    num_prompts_list = [10, 50, 100]
    concurrency_list = [1, 5, 10]

    all_results = []

    for complexity in complexities:
        for num_prompts in num_prompts_list:
            for concurrency in concurrency_list:
                result = run_experiment(complexity, num_prompts, concurrency)
                all_results.append(result)
                print(f"Complexity: {complexity}, Prompts: {num_prompts}, Concurrency: {concurrency}")
                print(f"Total Time: {result['total_time']:.2f}s, Avg Time: {result['avg_time']:.2f}s")
                print("---")

    # Analyze results
    df = pd.DataFrame(all_results)
    print(df[['complexity', 'num_prompts', 'concurrency', 'total_time', 'avg_time']])

    # You can add more detailed analysis here, such as plotting graphs

if __name__ == "__main__":
    main()