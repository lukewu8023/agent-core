# models/deepseek_coder.py

from .base_model import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class DeepSeekCoderModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_instance = ChatOpenAI(
            model_name="deepseek-coder", temperature=0.1, verbose=True
        )

    def name(self) -> str:
        return "deepseek-coder"
