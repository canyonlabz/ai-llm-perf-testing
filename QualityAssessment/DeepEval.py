import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, SummarizationMetric, LatencyMetric

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def test_llm_output():
    # Create a test case
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=["Paris is the capital and most populous city of France."],
        latency=0.5  # Simulated latency in seconds
    )

    # Define metrics
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    summarization_metric = SummarizationMetric(
        threshold=0.7,
        assessment_questions=[
            "Does the answer correctly state the capital of France?",
            "Is the answer concise and to the point?"
        ]
    )
    latency_metric = LatencyMetric(max_latency=1.0)

    # Run the test
    assert_test(test_case, metrics=[answer_relevancy_metric, summarization_metric, latency_metric])

if __name__ == "__main__":
    test_llm_output()