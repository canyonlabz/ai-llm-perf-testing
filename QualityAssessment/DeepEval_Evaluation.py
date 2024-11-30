import ollama
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

def evaluate_llm_output(user_input):
    # Use Ollama to generate a response
    response = ollama.chat(model='llama3.2:1b', messages=[
        {
            'role': 'user',
            'content': user_input
        }
    ])
    
    # Extract the generated text from the response
    llm_output = response['message']['content']

    # Create a test case
    test_case = LLMTestCase(
        input=user_input,
        actual_output=llm_output,
        retrieval_context=[user_input]  # Using input as context for simplicity
    )

    # Define the evaluation metric
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

    # Evaluate the test case
    result = assert_test(test_case, [answer_relevancy_metric])

    # Extract and return the score
    score = answer_relevancy_metric.score
    return llm_output, score

# Main execution
if __name__ == "__main__":
    user_input = input("Enter your prompt: ")
    llm_output, score = evaluate_llm_output(user_input)
    
    print(f"LLM Output: {llm_output}")
    print(f"DeepEval Score: {score}")