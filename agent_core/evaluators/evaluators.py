# evaluators/evaluators.py

from .coding_evaluator import CodingEvaluator
from .generic_evaluator import GenericEvaluator
from .toolcalling_evaluator import ToolCallingEvaluator


def get_evaluator(model_name):
    """
    Return a dict {category: evaluator_instance} tied to the given model instance.
    This is the default mapping used when an Agent is created.
    """
    return {
        "writing": GenericEvaluator(model_name),
        "summarization": GenericEvaluator(model_name),
        "action": GenericEvaluator(model_name),
        "coding": CodingEvaluator(model_name),
        "toolcalling": ToolCallingEvaluator(model_name),
        # You can add more categories here.
        # "default" can also be a fallback if desired:
        "default": GenericEvaluator(model_name),
    }
