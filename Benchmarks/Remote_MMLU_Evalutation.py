import requests
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

OLLAMA_API_URL = "http://remote-ollama-ip:11434/api/generate"  # Replace with your remote Ollama IP and port

# Load the MMLU dataset for a specific subject
def load_mmlu_data(subject):
    dataset = load_dataset("cais/mmlu", subject)
    return dataset["test"]

# Format the question and choices into a prompt
def format_prompt(question, choices):
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "\nPlease provide the correct answer as a single letter (A, B, C, or D)."
    return prompt

# Query the Ollama API with a prompt and return the model's response
def query_ollama(prompt, model="llama2:8b"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()['response'].strip().upper()
    else:
        raise Exception(f"Error querying Ollama: {response.text}")

# Evaluate the Ollama model on the MMLU dataset for a specific subject
def evaluate_model(subject):
    data = load_mmlu_data(subject)
    correct_answers = 0
    total_questions = len(data)

    for item in tqdm(data, desc=f"Evaluating {subject}"):
        question = item['question']
        choices = item['choices']
        correct_answer = item['answer']

        prompt = format_prompt(question, choices)
        model_answer = query_ollama(prompt)

        if model_answer == correct_answer:
            correct_answers += 1

    accuracy = correct_answers / total_questions
    return accuracy

# Main function to evaluate the Ollama model on multiple subjects
def main():
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]  # Add more subjects as needed

    scores = []
    for subject in subjects:
        score = evaluate_model(subject)
        scores.append(score)
        print(f"{subject}: {score:.2%}")

    overall_score = np.mean(scores)
    print(f"\nOverall MMLU Score: {overall_score:.2%}")

# Entry point of the script
if __name__ == "__main__":
    main()