import argparse
from langchain.evaluation import load_evaluator
from langchain.llms import OpenAI

def evaluate_response(response, reference):
    evaluator = load_evaluator("qa")
    llm = OpenAI(model_name="text-davinci-002")
    eval_result = evaluator.evaluate_strings(prediction=response, reference=reference)
    print(f"Evaluation result: {eval_result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=str, required=True)
    args = parser.parse_args()
    evaluate_response(args.response, "expected reference text")
