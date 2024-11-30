import ollama
from langchain.llms import Ollama
from langchain.evaluation import load_evaluator
from langchain_core.prompts import PromptTemplate

ollama_model = Ollama(model="llama3.1:8b")

def generate_response(prompt):
    return ollama_model.invoke(prompt)

evaluator = load_evaluator("criteria", criteria="relevance", llm=ollama_model)

def evaluate_response(prompt, response):
    eval_result = evaluator.evaluate_strings(
        prediction=response,
        input=prompt
    )
    return eval_result

def main():
    while True:
        user_prompt = input("Enter your prompt (or 'quit' to exit): ")
        if user_prompt.lower() == 'quit':
            break

        response = generate_response(user_prompt)
        evaluation = evaluate_response(user_prompt, response)

        print(f"Response: {response}")
        print(f"Evaluation Score: {evaluation['score']}")
        print(f"Evaluation Reasoning: {evaluation['reasoning']}")
        print("---")

if __name__ == "__main__":
    main()