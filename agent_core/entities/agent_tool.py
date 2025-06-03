from typing import List, Any
from langchain_core.tools import BaseTool
from mcp import ClientSession

from agent_core.protocols.mcp.mcp_server import MCPServer
import json


def tool_knowledge_format(tools: List[BaseTool] = None) -> str:
    tools_knowledge_list = []
    if tools:
        tools_knowledge_list = [
            str(tool.args_schema.model_json_schema()) for tool in tools
        ]
    tools_knowledge = "\n".join(tools_knowledge_list)
    return tools_knowledge


class AgentTool:

    def __init__(self, langchain_tools: List[BaseTool] = list(), mcp_servers: List[MCPServer] = list()):
        self.langchain_tools = langchain_tools
        self.mcp_servers = mcp_servers
        self.agent_tool = {}
        self.tool_type = {}
        self.mcp_servers_map = {}
        self.langchain_tool_map = {}

    async def get_tool(self):
        if self.langchain_tools:
            for langchain_tool in self.langchain_tools:
                self.agent_tool[langchain_tool.name] = langchain_tool.args_schema.model_json_schema()
                self.tool_type[langchain_tool.name] = "langchain"
                self.langchain_tool_map[langchain_tool.name] = langchain_tool
        if self.mcp_servers:
            for mcp_server in self.mcp_servers:
                session = await mcp_server.connect()
                async with session:
                    mcp_tools = await mcp_server.get_tools()
                    for mcp_tool in mcp_tools:
                        schema = mcp_tool.inputSchema
                        schema["description"] = mcp_tool.description
                        self.agent_tool[mcp_tool.name] = schema
                        self.tool_type[mcp_tool.name] = "mcp"
                        self.mcp_servers_map[mcp_tool.name] = mcp_server

    def get_tool_knowledge(self):
        return "\n".join(
            f"tool name: {k}, schema: {json.dumps(v)}" for k, v in self.agent_tool.items()
        )

    def get_tool_description(self, name: str):
        if self.tool_type[name] == "langchain":
            return self.langchain_tool_map[name].description
        return self.agent_tool[name]["description"]

    def get_tool_schema(self, name: str):
        return self.agent_tool[name]

    async def execute_tool(self, name: str, arg: Any = None):
        if self.tool_type[name] == "langchain":
            return self.langchain_tool_map[name].invoke(arg)
        server = self.mcp_servers_map[name]
        await server.connect()
        return await server.tool_calling(name, arg)

