from langchain.llms import Ollama
from langchain.evaluation import load_evaluator
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.prompts import PromptTemplate
import time

# Set up Ollama client
ollama = Ollama(base_url="http://localhost:11434", model="llama3.1:8b")

# Define evaluation criteria
EVAL_CRITERIA = {
    "relevance": "Does the response directly address the prompt?",
    "coherence": "Is the response well-structured and logically consistent?",
    "factual_accuracy": "Does the response contain accurate information?",
    "helpfulness": "Is the response helpful in addressing the user's needs?",
    "creativity": "Does the response demonstrate creative thinking?",
    "conciseness": "Is the response concise and to the point?",
    "language_quality": "Is the language used in the response of high quality?",
}

# Function to generate response using Ollama
def generate_response(prompt):
    return ollama(prompt)

# Function to evaluate response using different Langchain Evaluators
def evaluate_response(prompt, response):
    results = {}
    
    # Criteria-based evaluation
    for criterion, description in EVAL_CRITERIA.items():
        evaluator = load_evaluator("criteria", criteria=criterion)
        eval_result = evaluator.evaluate_strings(prediction=response, input=prompt)
        results[criterion] = {
            "score": eval_result["score"],
            "reasoning": eval_result["reasoning"]
        }
    
    return results

# Main function to run the evaluation
def run_evaluation():
    prompts = [
        "Explain the concept of quantum entanglement.",
        "What are the ethical implications of artificial intelligence?",
        "How can we address climate change on a global scale?",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        start_time = time.time()
        response = generate_response(prompt)
        generation_time = time.time() - start_time
        
        print(f"Response: {response}")
        print(f"Generation Time: {generation_time:.2f} seconds")
        
        evaluation_results = evaluate_response(prompt, response)
        
        print("\nEvaluation Results:")
        for criterion, result in evaluation_results.items():
            print(f"{criterion.capitalize()}:")
            print(f"  Score: {result['score']}")
            print(f"  Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    run_evaluation()