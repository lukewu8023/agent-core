import asyncio
from typing import Optional, Any
from mcp import ClientSession, Tool
from mcp.client.streamable_http import streamablehttp_client
from contextlib import AsyncExitStack, asynccontextmanager


class MCPServer:

    url: str
    type: Optional[str] = "streamable-http"
    headers: Optional[dict[str, str]]

    def __init__(self, url: str):
        self.url = url
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    async def connect(self):
        try:
            transport = await self.exit_stack.enter_async_context(self.create_streams())
            read, write, _ = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(
                    read,
                    write
                )
            )
            await session.initialize()
            self.session = session
            return session
        except Exception:
            await self.cleanup()
            raise

    async def tool_calling(self, tool_name: str, arguments: dict[str, Any] | None):
        result = await self.session.call_tool(tool_name, arguments)
        return result.content[0].text

    async def get_tools(self) -> list[Tool]:
        return (await self.session.list_tools()).tools

    def create_streams(
        self,
    ):
        """Create the streams for the server."""
        return streamablehttp_client(
            url=self.url
        )

    async def cleanup(self):
        """Cleanup the server."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
            except Exception:
                raise
            finally:
                self.session = None
