from pydantic import BaseModel, RootModel
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os
from pathlib import Path


# Set up a persistent cache for LLM calls
if 'PROJECT_ROOT' not in os.environ:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')

PROJECT_ROOT = os.getenv('PROJECT_ROOT')
database_path = os.path.join(PROJECT_ROOT, 'data', '.llm_cache.db')
set_llm_cache(SQLiteCache(database_path=database_path))


class StructuredLLM():
    def __init__(self, model: str, response_model: BaseModel | RootModel):
        self.model = model
        self.response_model = response_model
        self.chat_model = init_chat_model(model).with_structured_output(
            response_model, include_raw=True)

    def invoke(self, prompt: str, **kwargs):
        """
        Light wrapper for model's .invoke() method

        Args:
            prompt (str): The fully formatted system prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model's invoke method
        Returns:
            The model's response object, which includes both parsed and raw outputs
        """
        messages = [SystemMessage(content=prompt)]

        # 1-23-25: Claude models require a non-empty human message to respond, or you get a Bad Request error
        if self.model.startswith("claude") or self.model.startswith("anthropic"):
            messages += [HumanMessage(content="do it")]
        response = self.chat_model.invoke(input=messages, **kwargs)
        return response


def main():
    print("Hello from structured-output-llms!")


if __name__ == "__main__":
    main()
