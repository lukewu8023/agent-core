# models/gemini_15_flash_002.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os


class Gemini15Flash002Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="gemini-1.5-flash-002", temperature=0.1, verbose=True
        )

    def name(self) -> str:
        return "gemini-1.5-flash-002"

