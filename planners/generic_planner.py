# planners/generic_planner.py

import json
from typing import List, Optional

from langchain_core.tools import BaseTool

from models.model_registry import ModelRegistry
from utils.logger import get_logger
from config import Config


class Step:
    """
    Represents a step with a name and a description.
    Now includes 'category' so we can determine which validator to use.
    """

    def __init__(
        self,
        name: str,
        description: str,
        use_tool: bool,
        tool_name: str,
        category: str = "default",
    ):
        self.name = name
        self.description = description
        self.use_tool = use_tool
        self.tool_name = tool_name
        self.category = category

    def __repr__(self):
        return (
            f"Step(name='{self.name}', description='{self.description}', "
            f"use_tool='{self.use_tool}', tool_name='{self.tool_name}', category='{self.category}')"
        )

    def to_dict(self):
        # Convert the object to a dictionary that can be serialized to JSON
        return {
            "name": self.name,
            "description": self.description,
            "use_tool": self.use_tool,
            "tool_name": self.tool_name,
            "category": self.category,
        }


class GenericPlanner:
    """
    A simple planner that calls the model to break a task into JSON steps.
    Each step may optionally specify a category, used for specialized validation.
    """

    DEFAULT_PROMPT = """\
The possible categories for each step are: {categories_str}, try to categorize step to one of these categories, if not, define new category and put in step_category.
    
Given the following task and the possible tools, generate a plan based on provided knowledge by breaking it down into actionable steps.
Present each step in JSON format with the attributes 'step_name', 'step_description', 'use_tool', and optionally 'tool_name', and 'step_category'.
And if need to use the tool, please make sure 'step_description' should contain tool's properties needed information
All steps should be encapsulated under the 'steps' key.

<Knowledge>
{knowledge}
</Knowledge>

<Examples>
{example_json1}

{example_json2}
</Examples>

<Tools>
{tools_knowledge}
</Tools>

<Task>
{task}
</Task>

Output ONLY valid JSON. No extra text or markdown.
Steps:
"""

    EXAMPLE_JSON1 = """\
{
    "steps": [
        {
            "step_name": "Prepare eggs",
            "step_description": "Get the eggs from the fridge and put them on the table.",
            "use_tool": true,
            "tool_name": "Event",
            "step_category": "action"
        },
        ...
    ]
}"""

    EXAMPLE_JSON2 = """\
{
    "steps": [
        {
            "step_name": "Plan code structure",
            "step_description": "Outline the classes and methods.",
            "use_tool": false,
            "step_category": "coding"
        },
        ...
    ]
}"""

    def __init__(
        self, model: str = None, log_level: Optional[str] = None, prompt: str = None
    ):
        """
        If 'model' is not provided, the default model from config will be used.
        'prompt' can override the default prompt used for planning.
        """
        self.logger = get_logger("generic-planner", log_level)
        if not model:
            model = Config.DEFAULT_MODEL

        self.model_name = model
        self.model = ModelRegistry.get_model(self.model_name)
        if not self.model:
            self.logger.error(f"Model '{self.model_name}' not found for planning.")
            raise ValueError(
                f"Model '{self.model_name}' is not supported for planning."
            )
        self.logger.info(f"GenericPlanner initialized with model: {self.model.name}")

        self._prompt = prompt or self.DEFAULT_PROMPT

    @property
    def prompt(self) -> str:
        """Get or set the single plan prompt (for dividing tasks into steps)."""
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    def plan(
        self,
        task: str,
        tools: Optional[List[BaseTool]],
        execute_history: list = None,
        knowledge: str = "",
        background: str = "",
        categories: Optional[List[str]] = None,
        agent=None,
    ) -> List[Step]:
        """
        Use the LLM to break down the task into multiple steps in JSON format.
        'knowledge' is appended to the prompt to guide the planning process.
        If 'categories' is provided, we pass it to the LLM so it can properly categorize each step.
        """
        self.logger.info(f"Creating plan for task: {task}")
        tools_knowledge_list = []
        if tools is not None:
            tools_knowledge_list = [
                str(tool.args_schema.model_json_schema()) for tool in tools
            ]
        tools_knowledge_str = "\n".join(tools_knowledge_list)

        if categories:
            categories_str = ", ".join(categories)
        else:
            categories_str = "(Not defined)"

        final_prompt = self._prompt.format(
            knowledge=knowledge,
            task=task,
            tools_knowledge=tools_knowledge_str,
            example_json1=self.EXAMPLE_JSON1,
            example_json2=self.EXAMPLE_JSON2,
            categories_str=categories_str,
        )

        response = self.model.process(final_prompt)
        response_text = str(response)

        if not response_text or not response_text.strip():
            self.logger.error("LLM returned an empty or null response.")
            raise ValueError("LLM returned an empty or null response.")

        self.logger.debug(f"Raw LLM response: {repr(response_text)}")

        # Minor cleanup of possible code fences
        cleaned = response_text.replace("```json", "").replace("```", "").strip()

        # Attempt JSON parse
        try:
            data = json.loads(cleaned)
            steps_data = data.get("steps", [])
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw LLM response was: {cleaned}")
            raise ValueError("Invalid JSON format in planner response.")

        valid_categories = set(categories) if categories else set()

        # Convert steps_data to Step objects
        results = []
        for sd in steps_data:
            name = sd.get("step_name")
            desc = sd.get("step_description")
            use_tool = sd.get("use_tool")
            tool_name = sd.get("tool_name")
            raw_cat = sd.get("step_category", "default")

            # If LLM returned a category not in recognized set, fallback to 'default'
            final_cat = raw_cat if raw_cat in valid_categories else "default"

            if name and desc:
                results.append(
                    Step(
                        name=name,
                        description=desc,
                        use_tool=use_tool,
                        tool_name=tool_name,
                        category=final_cat,
                    )
                )
            else:
                self.logger.warning(f"Incomplete step data: {sd}")

        self.logger.info(f"Got {len(results)} steps from the LLM.")
        return results

    def parse_steps(self, response: str) -> List[Step]:
        """
        Parse the LLM response to extract individual steps and return them as Step objects.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            List[Step]: A list of Step objects extracted from the response.
        """
        self.logger.debug(f"Raw planner response: {response}")
        try:
            data = json.loads(response)
            steps_data = data.get("steps", [])
            steps = []
            for step in steps_data:
                step_name = step.get("step_name")
                step_description = step.get("step_description")
                if step_name and step_description:
                    steps.append(
                        Step(
                            name=step_name,
                            description=step_description,
                            category=step.get("step_category", "default"),
                        )
                    )
                else:
                    self.logger.warning(f"Incomplete step data: {step}")
            return steps
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError("Invalid JSON format received from the planner.")
        except Exception as e:
            self.logger.error(f"Unexpected error during parsing steps: {e}")
            raise
