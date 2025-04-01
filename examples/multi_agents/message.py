from dataclasses import dataclass

@dataclass
class Question:
    content: str

@dataclass
class Answer:
    content: str

@dataclass
class Request:
    content: str
    question: str

@dataclass
class IntermediateResponse:
    content: str
    question: str
    answer: str
    round: int
    sender: str

@dataclass
class FinalResponse:
    answer: str
    sender: str
