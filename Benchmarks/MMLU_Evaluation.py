import ollama
import numpy as np
import csv
from datasets import load_dataset
from tqdm import tqdm

# Load the MMLU dataset for a specific subject
def load_mmlu_data(subject):
    dataset = load_dataset("cais/mmlu", subject)
    return dataset["test"]

# Format the question and choices into a prompt for the Ollama model
def format_prompt(question, choices):
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "\nProvide ONLY a single letter (A, B, C, or D) as your answer. Do not explain or elaborate.\n"
    return prompt

# Evaluate the Ollama model on the MMLU dataset for a specific subject
def evaluate_model(subject, csv_writer):
    data = load_mmlu_data(subject)
    correct_answers = 0
    total_questions = len(data)

    for item in tqdm(data, desc=f"Evaluating {subject}"):
        question = item['question']
        choices = item['choices']
        correct_answer = chr(65 + item['answer'])  # Convert 0-3 to A-D

        prompt = format_prompt(question, choices)
        response = ollama.chat(model='llama3.2:1b', messages=[
            {'role': 'user', 'content': prompt}
        ])

        # Extract model answer and log to CSV
        model_answer = response['message']['content'].strip().upper() if 'message' in response and 'content' in response['message'] else "N/A"
        
        # Write to CSV for troubleshooting
        csv_writer.writerow([question, choices, correct_answer, model_answer])        

        # Check if the model answer is correct
        if model_answer == correct_answer:
            correct_answers += 1

    accuracy = correct_answers / total_questions
    return accuracy

def main():
    # Prepare CSV file for logging
    with open('mmlu_evaluation_log.csv', mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Question", "Choices", "Correct Answer", "Model Response"])  # CSV headers

        subjects = ["college_computer_science"]
        
        scores = []
        for subject in subjects:
            # Pass csv_writer into evaluate_model to ensure each subject logs correctly
            score = evaluate_model(subject, csv_writer)
            scores.append(score)
            print(f"{subject}: {score:.2%}")

        overall_score = np.mean(scores)
        print(f"\nOverall MMLU Score: {overall_score:.2%}")

# Run the main function
if __name__ == "__main__":
    main()