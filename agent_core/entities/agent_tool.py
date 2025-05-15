from typing import List, Any
from langchain_core.tools import BaseTool
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

    def __init__(self, langchain_tools: List[BaseTool] = None, mcp_servers: List[MCPServer] = None):
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
        for mcp_server in self.mcp_servers:
            try:
                mcp_tool_knowledge = await mcp_server.get_tools()
                print(f"[AgentTool] Got {len(mcp_tool_knowledge)} tools from MCP server {mcp_server.url}")
                for tool_knowledge in mcp_tool_knowledge:
                    schema = tool_knowledge.inputSchema
                    schema["description"] = tool_knowledge.description
                    self.agent_tool[tool_knowledge.name] = schema
                    self.tool_type[tool_knowledge.name] = "mcp"
                    self.mcp_servers_map[tool_knowledge.name] = mcp_server
            except Exception as e:
                print(f"[AgentTool] Failed to get tools from {mcp_server.url}: {e}")

    def get_tool_knowledge(self):
        return "\n".join(
            f"tool name: {k}, schema: {json.dumps(v)}" for k, v in self.agent_tool.items()
        )

    def get_tool_description(self, name: str):
        if self.tool_type[name] == "langchain":
            return self.langchain_tool_map[name].description
        return self.agent_tool[name]["description"]

    async def execute_tool(self, name: str, arg: Any = None):
        if self.tool_type[name] == "langchain":
            return self.langchain_tool_map[name].invoke(arg)
        server = self.mcp_servers_map[name]
        return await server.tool_calling(name, arg)
