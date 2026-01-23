# Structured LLM Outputs

This repo provides the `StructuredLLM` class, which provides a common interface over OpenAI, Anthropic, Ollama, and other LLM provides to produce structured outputs. A `StructuredLLM` instance is intended to work the same as a standard LangChain chat model, but instantiated with a Pydantic model to define response structure, regardless of the LLM provider used.
