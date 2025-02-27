# examples/autogen_integration.py

import sys
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agent_core.agents.autogen_agent import PhaenixAgent
from autogen_agentchat.ui import Console

from agent_core.planners import GraphPlanner

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import asyncio

async def main():

    # case 1: build minimal PhaenixAgent
    agent = PhaenixAgent("AutogenAgent", "gemini-1.5-flash-002")
    await Console(agent.run_stream(task="who are you"))

    # case 2: build advanced PhaenixAgent
    advanced_agent = PhaenixAgent(
        name="AdvancedAutogenAgent",
        model_name="gemini-1.5-flash-002",
        planner=GraphPlanner(),
        evaluators_enabled=True,
        knowledge="To draw a object you need to take 3 steps, 1) prepare tools, 2) prepare "
                     "paper, 3) draw the object",
        background="You are a professional artist")

    task = "draw a flower"
    await Console(advanced_agent.run_stream(task=task))

    # case 3: group chat between PhaenixAgent and Autogen build-in agent(AssistantAgent)
    reviewer_agent = AssistantAgent(
        "Reviewer",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", base_url="https://api.ohmygpt.com/v1/"),
        system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedbacks are addressed.",
    )

    advanced_agent = PhaenixAgent(
        name="AdvancedAutogenAgent",
        model_name="gemini-1.5-flash-002",
        planner=GraphPlanner(),
        evaluators_enabled=True)

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat([advanced_agent, reviewer_agent], termination_condition=text_termination)

    await Console(team.run_stream(task="Write a short poem about the fall season with Chinese Tang-style."))

if __name__ == "__main__":
    asyncio.run(main())