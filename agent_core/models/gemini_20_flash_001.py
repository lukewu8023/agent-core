# models/gemini_15_flash_002.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI


class Gemini15Flash002Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="gemini-2.0-flash-001", temperature=0.1, verbose=True
        )

    def name(self) -> str:
        return "gemini-2.0-flash-001"

