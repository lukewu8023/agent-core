import asyncio
from agent_core.agents.supervisor_agent import SuperVisorAgent
from agent_core.protocols.a2a.server import A2AServer
from agent_core.protocols.a2a.server.request_handlers import DefaultA2ARequestHandler
from agent_core.protocols.a2a.types import AgentCapabilities, AgentSkill, AgentCard, AgentAuthentication
from fastapi import FastAPI, WebSocket
import uvicorn

# app = FastAPI()
# supervisor_agent = SuperVisorAgent(['http://localhost:8882'])
# capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
# supervisor_skill = AgentSkill(
#     id="supervisor_agent_skill",
#     name="supervisor agent skill",
#     description="Answer user questions based on knowledge. Look up information for issue troubleshooting",
#     tags=["Q & A", "answer question", "troubleshooting"],
#     examples=["What is github?", "Why cpu usage is high?"],
# )
# supervisor_agent_card = AgentCard(
#     name="supervisor Agent",
#     description="Helps with answer user query based on knowledge and issue troubleshooting",
#     url=f"http://localhost:8880/",
#     version="1.0.0",
#     defaultInputModes=["text", "text/plain"],
#     defaultOutputModes=["text", "text/plain"],
#     capabilities=capabilities,
#     skills=[supervisor_skill],
#     authentication=AgentAuthentication(schemes=['public'])
# )
#
# supervisor_agent_server = A2AServer(
#     agent_card=supervisor_agent_card,
#     request_handler=DefaultA2ARequestHandler(
#         agent_executor=supervisor_agent
#     )
# )
#
# supervisor_agent_server.start(host='localhost', port=8880)
#
# async def process(ws):
#     supervisor_agent = SuperVisorAgent(['http://localhost:8882'])
#     await supervisor_agent.execute_ws("What is EMC?", ws)
#     await supervisor_agent.execute_ws(
#         "Find the specifics root cause and get more detail about why the event id: 10000 in IE component failed?", ws)
#     async for message in ws:
#         await supervisor_agent.execute_ws(message, ws)

app = FastAPI()


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = SuperVisorAgent(['http://localhost:8882', 'http://localhost:8881'])

    while True:
        data = await websocket.receive_text()
        await agent.execute_ws(data, websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9090, reload=False)

