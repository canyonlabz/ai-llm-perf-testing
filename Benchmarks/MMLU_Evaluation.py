import ollama
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def load_mmlu_data(subject):
    dataset = load_dataset("cais/mmlu", subject)
    return dataset["test"]

def format_prompt(question, choices):
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "\nPlease provide the correct answer as a single letter (A, B, C, or D)."
    return prompt

def evaluate_model(subject):
    data = load_mmlu_data(subject)
    correct_answers = 0
    total_questions = len(data)

    for item in tqdm(data, desc=f"Evaluating {subject}"):
        question = item['question']
        choices = item['choices']
        correct_answer = item['answer']

        prompt = format_prompt(question, choices)
        response = ollama.chat(model='llama3.2:1b', messages=[
            {'role': 'user', 'content': prompt}
        ])

        model_answer = response['message']['content'].strip().upper()
        if model_answer == correct_answer:
            correct_answers += 1

    accuracy = correct_answers / total_questions
    return accuracy

def main():
    '''Evaluate the MMLU benchmark using the Ollama model.
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", 
                "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", 
                "college_medicine", "college_physics", "computer_security", "conceptual_physics", 
                "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", 
                "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", 
                "high_school_european_history", "high_school_geography", "high_school_government_and_politics", 
                "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", 
                "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", 
                "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", 
                "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", 
                "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", 
                "prehistory", "professional_accounting", "professional_law", "professional_medicine", 
                "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", 
                "virology", "world_religions"]
    '''
    subjects = ["college_computer_science", "college_mathematics", "college_medicine", "college_physics","machine_learning"]    
                
    scores = []
    for subject in subjects:
        score = evaluate_model(subject)
        scores.append(score)
        print(f"{subject}: {score:.2%}")

    overall_score = np.mean(scores)
    print(f"\nOverall MMLU Score: {overall_score:.2%}")

if __name__ == "__main__":
    main()