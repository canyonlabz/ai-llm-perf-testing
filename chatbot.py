from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below:

Here is the conversation history: {conversation_history}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3.1:8b", device="cuda")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    conversation_history = ""
    print("Welcome to the chatbot! Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        result = chain.invoke({"conversation_history": conversation_history, "question": question})
        print("Bot:", result)
        conversation_history += f"Bot: {result}\nYou: {question}\n"

if __name__ == "__main__":
    handle_conversation()


