# models/gemini_15_pro_002.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os


class Gemini15PRO002Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="gemini-1.5-pro-002", temperature=0.1, verbose=True
        )

    def process(self, request: str) -> str:
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
        return "gemini-1.5-pro-002"
