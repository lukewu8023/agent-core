# models/gpt_4o_mini.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI


class GPT4OMiniModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="gpt-4o-mini", temperature=0.1, verbose=True
        )

    def name(self) -> str:
        return "gpt-4o-mini"
