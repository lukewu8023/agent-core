# examples/override_default_prompts.py

import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agent_core.agents import Agent
from agent_core.planners import GenericPlanner
from agent_core.evaluators import GenericEvaluator


def main():
    """
    Demonstrates how to override default prompts for:
    1) evaluator (Scoreevaluator)
    2) Agent single-step execution
    3) Planner multistep prompt
    4) Agent's final summary prompt

    These overrides only apply to the current instances of evaluator, agent, and planner.
    Once a new object is created, it will revert to the default prompt.
    """

    # --- 1) OVERRIDE Scoreevaluator Prompt ----------------------------------
    evaluator = GenericEvaluator()
    # Print current default evaluator prompt
    print("Default evaluator Prompt:\n", evaluator.prompt)

    # Replace it with a grammar-focused prompt; note the placeholders {request} and {response} must remain
    evaluator.prompt = """\
    You are a special evaluator focusing on only grammar correctness.
    Subtask: {request}
    Output: {response}
    Please check grammar only.
    """

    # --- 2) OVERRIDE Agent Execution Prompt ---------------------------------
    agent = Agent()

    # Print the Agent's default single-step execution prompt
    print("Default Agent Execution Prompt:\n", agent.execute_prompt)

    # Override the agent's single-step prompt
    # Valid placeholders here are {task} and {background} (and {context_section} if you need it).
    agent.execute_prompt = (
        "Customized single-step prompt.\n" "Task: {task}\n" "Background: {background}\n"
    )

    # Provide some background context for the agent (this is optional, but demonstrates usage)
    agent.background = "I am a helpful assistant."

    # --- 3) OVERRIDE GenericPlanner Prompt -----------------------------------
    planner = GenericPlanner()

    # Print the default planner prompt
    print("Default Planner Prompt:\n", planner.prompt)

    # Override the planner's prompt to produce exactly two steps in JSON
    # Placeholders typically include: {task}, {tools_knowledge}, {knowledge}, {example_json1}, {example_json2}.
    planner.prompt = """\
    Given the following task and the tools, generate a high-level plan by breaking it down into meaningful, actionable steps.

    **Instructions for generating 'use_tool':**
    If the <Tools></Tools> section is empty, set "use_tool" to false for all steps and omit "tool_name."
    If the <Tools></Tools> section contains tools, set "use_tool" to true when a tool is necessary. Include "tool_name" in those steps and reference any tool-specific properties or arguments in the description.

    **Task Breakdown Requirements:**
    1) All steps must be encapsulated under the "steps" key in valid JSON format.
    2) Each step should include:
        "step_name": The name of the step
        "step_description": A concise description of the action to be performed in that step
        "use_tool": A boolean indicating whether a tool should be used
        Optionally, "tool_name": The name of the tool if "use_tool" is true
        "step_category": Categorize the step based on its function ({categories_str})
    3) The possible categories for each step are: {categories_str}.
      If you cannot fit into any existing category, define a new category in "step_category".
    4) Output **ONLY** valid JSON. No extra text, no Markdown.   
    5) Steps should be high-level but clear and not missing any aspect, had better used all tools to analyse, avoiding overly detailed breakdowns for simple tasks.
    
    {background}
    
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
    
    I only want two simple steps.
    Output ONLY valid JSON. No extra text or markdown.
    Steps:
    """

    # Attach this planner to the agent
    agent.planner = planner

    # Now execute a simple task that will be broken into steps by the new planner prompt
    result = agent.execute("draw a flower")
    print("Execution result:", result)

    # --- 4) OVERRIDE Agent Summary Prompt ------------------------------------
    # Print the current default summary prompt
    print("Default Agent Summary Prompt:\n", agent.summary_prompt)

    # Replace it with a custom summary format
    # The placeholder {history_text} is used to inject the step-by-step details.
    agent.summary_prompt = "Here is a custom summary for steps:\n{history_text}"

    # Retrieve and print the final execution summary
    print(agent.get_execution_result_summary())


if __name__ == "__main__":
    main()
