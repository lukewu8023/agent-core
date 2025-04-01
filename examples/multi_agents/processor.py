import asyncio

from agent_core.agents import Agent
from autogen_core import (
    SingleThreadedAgentRuntime,
    TypeSubscription, TopicId
)


from worker import Worker
from supervisor import AgentAggregator
from message import Question

async def main():


    qaAgent = Agent()
    qaAgent.background = "you are professional to answer user's question based on your knowledge"

    issueAgent = Agent()
    issueAgent.background = "you are good at troubleshooting"

    task = """
    how to clone repo from github
    """

    runtime = SingleThreadedAgentRuntime()

    await Worker.register(runtime, "QAAgent", lambda: Worker(source_agent=qaAgent, topic_type="QAAgent", max_round=2))
    await Worker.register(runtime, "issueAgent", lambda: Worker(source_agent=issueAgent, topic_type="issueAgent", max_round=1))
    await runtime.add_subscription(TypeSubscription(f"QAAgent", f"QAAgent"))
    await runtime.add_subscription(TypeSubscription(f"issueAgent", f"issueAgent"))

    await AgentAggregator.register(runtime, "SuperVisorAgent", lambda: AgentAggregator(target_topic="QAAgent", second_target_topic="issueAgent"))

    runtime.start()
    await runtime.publish_message(Question(content=task), TopicId("SuperVisorAgent", source="User"))
    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())