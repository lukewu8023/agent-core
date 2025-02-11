# models/deepseek_reasoner.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class DeepSeekReasonerModel(BaseModel):
    def __init__(self):
        super().__init__(name="deepseek-reasoner")
        self.model_instance = ChatOpenAI(
            model="deepseek-reasoner", temperature=0.1, verbose=True
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
