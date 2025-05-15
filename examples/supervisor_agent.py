import asyncio
from time import sleep

from agent_core.agents.supervisor_agent import SuperVisorAgent


async def main():
    supervisor_agent = SuperVisorAgent(['http://localhost:8880'])
    await supervisor_agent.execute("What is EMC?")
    # await supervisor_agent.execute("Find the specifics root cause and get more detail about why the event id: 10000 in IE component failed?")

if __name__ == "__main__":
    asyncio.run(main())
