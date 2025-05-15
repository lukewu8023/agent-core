# models/gpt_35_turbo.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI


class GPT35TURBOModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.1, verbose=True
        )

    def name(self) -> str:
        return "gpt-3.5-turbo"
