# models/deepseek_reasoner.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class DeepSeekReasonerModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="deepseek-reasoner", temperature=0.1, verbose=True
        )

    def invoke(self, request: str) -> str:
        messages = [
            HumanMessage(request),
        ]
        response = self.model_instance.invoke(messages)
        # Extract the 'content' attribute to return a string
        if hasattr(response, "content"):
            self.add_token(response)
            return response.content
        else:
            # Fallback in case 'content' is missing
            return str(response)

    def name(self) -> str:
        return "deepseek-reasoner"
