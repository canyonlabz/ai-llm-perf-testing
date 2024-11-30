# ai-llm-perf-testing
Project for performance testing Large Language Models (LLMs) using Ollama as a proof-of-concept.

## Prerequisites:
1. Large Language Model: Ollama
2. Embeddings: Langchain
3. Vector Database:  ChromaDB
    * Prerequisite: Install Microsoft C++ Build Tools
        * Step 1: Download the installer from https://visualstudio.microsoft.com/visual-cpp-build-tools/
        * Step 2: During installation, select the following:
            - MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
            - Windows 11 SDK (10.0.22621.0)
    * Installation: Install ChrombaDB
        * Step 1: Read the documenation from https://docs.trychroma.com/
        * Step 2: pip install chromadb