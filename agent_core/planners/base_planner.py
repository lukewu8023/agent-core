# planners/base_planner.py

from abc import abstractmethod
from typing import List, Optional
from langchain_core.tools import BaseTool
from agent_core.agent_basic import AgentBasic
from agent_core.entities.steps import Steps


def tool_knowledge_format(tools: Optional[List[BaseTool]]) -> str:
    tools_knowledge_list = []
    if tools is not None:
        tools_knowledge_list = [
            str(tool.args_schema.model_json_schema()) for tool in tools
        ]
    tools_knowledge = "\n".join(tools_knowledge_list)
    return tools_knowledge


class BasePlanner(AgentBasic):
    """
    An abstract base class for all planners.
    Both GenericPlanner and GraphPlanner will inherit from this.
    """

    DEFAULT_PROMPT = """ 
    Given the following task and the tools, generate a high-level plan by breaking it down into meaningful, actionable steps.
    
    **Instructions for generating 'use_tool'**
    If the <Tools></Tools> section is empty, set "use_tool" to false for all steps and omit "tool_name."
    If the <Tools></Tools> section contains tools, set "use_tool" to true when a tool is necessary. Include "tool_name" in those steps and reference any tool-specific properties or arguments in the description.
    
    **Task Breakdown Requirements**
    1) All steps must be encapsulated under the "steps" key in valid JSON format.
    2) Each step should include:
        "name": The name of the step
        "description": A concise description of the action to be performed in that step
        "use_tool": A boolean indicating whether a tool should be used
        Optionally, "tool_name": The name of the tool if "use_tool" is true
        "category": Categorize the step based on its function ({categories_str})
    3) The possible categories for each step are: {categories_str}.
        If you cannot fit into any existing category, define a new category in "step_category".
    4) Steps should be high-level but clear and not missing any aspect, had better used all tools to analyse, avoiding overly detailed breakdowns for simple tasks.
    
    **Background**
    {background}
    
    **Knowledge**
    {knowledge}
    
    **Tools**
    {tools_knowledge}
    
    **Task**
    {task}
    
    **Examples**
    {example_json1}
    {example_json2}
    
    **Note:** Ensure your response is valid JSON, without any additional text or comments.
    Steps:
    """

    def __init__(
        self, model_name: Optional[str] = None, log_level: Optional[str] = None
    ):
        """
        Pass in the agent's model instance so we can call model.process(...) for evaluation prompts.
        Optionally specify log_level for debug or other logs.
        'prompt' can override the default prompt template.
        """
        super().__init__(self.__class__.__name__, model_name, log_level)
        self.prompt = self.DEFAULT_PROMPT

    @abstractmethod
    def plan(
        self,
        task: str,
        tools: Optional[List[BaseTool]],
        knowledge: str = "",
        background: str = "",
        categories: Optional[List[str]] = None,
    ) -> Steps:
        """
        Generates a plan (list of Steps, or node-based structure, etc.) from the LLM.
        """
        pass

    @abstractmethod
    def execute_plan(
        self,
        *args,  # used to allow flexible signatures in child classes
        **kwargs,
    ):
        """
        Executes the already-generated plan (steps or node graph).
        """
        pass
