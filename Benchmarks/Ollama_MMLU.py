from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

# Define a model class for Ollama LLM
class OllamaModel:
    def __init__(self, model_name="llama3.1:8b", device="cuda"):
        self.model = OllamaLLM(model=model_name, device=device)

    # This method must return a single string as an answer
    def generate(self, prompt):
        result = self.model.invoke({"prompt": prompt})
        return result.strip()

# Initialize your Ollama model
ollama_model = OllamaModel()

# Define the benchmark with specific tasks and modes
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

# Evaluate the Ollama model
try:
    results = benchmark.evaluate(ollama_model)
    if isinstance(results, float):
        print(f"Overall TruthfulQA Accuracy: {results}")
    else:
        print(f"Overall Score: {results.overall_score}")
except Exception as e:
    print(f"An error occurred during evaluation: {e}")
