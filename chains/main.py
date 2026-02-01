import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from pathlib import Path


# Set up a persistent cache for LLM calls
path = Path(__file__).resolve()
for parent in path.parents:
    env_file = parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def main():
    llm = ChatOpenAI(temperature=0)
    prompt_extract = ChatPromptTemplate.from_template(
        "Extract the the technical specifications from the following text:\n\n{text_input}")
    prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}")
    extraction_chain = prompt_extract | llm | StrOutputParser()
    full_chain = (
        {'specifications': extraction_chain}
        | prompt_transform
        | llm
        | StrOutputParser()
    )

    input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
    final_result = full_chain.invoke({'text_input': input_text})
    print("---JSON output---")
    print(final_result)


main()
