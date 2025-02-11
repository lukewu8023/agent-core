# models/gpt_35_turbo.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class GPT35TURBOModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.1, verbose=True
        )

    def process(self, request: str) -> str:
        messages = [
            HumanMessage(request),
        ]
        response = self.model_instance.invoke(messages)

        # Extract the 'content' attribute to return a string
        if hasattr(response, "content"):
            return response.content
        else:
            # Fallback in case 'content' is missing
            return str(response)

    def name(self) -> str:
        return "gpt-3.5-turbo"