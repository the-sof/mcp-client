import asyncio
import sys
import logging
import aiohttp
import json
import os
import re

from typing import Optional, Union
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/mcp_client.log"),
        logging.StreamHandler()  # Keep console output as well
    ]
)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        # self.http_session = None
        # self.http_transport = None

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an SSE MCP server.
        
        Args:
            server_url (str): URL of the SSE MCP server.
        """
        logger.debug(f"Connecting to SSE MCP server at {server_url}")

        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"\n\nConnected to SSE MCP Server at {server_url}. \nAvailable tools: {[tool.name for tool in tools]}")

    async def connect_to_stdio_server(self, server_script_path: str):
        """Connect to a stdio MCP server.
        
        Args:
            server_script_path (str): Path to the server script (.py, .js, or npm package).
        """
        is_python = False
        is_javascript = False
        command = None
        args = [server_script_path]
        
        # Determine if the server is a file path or npm package
        if server_script_path.startswith("@") or "/" not in server_script_path:
            # Assume it's an npm package
            is_javascript = True
            command = "npx"
        else:
            # It's a file path
            is_python = server_script_path.endswith(".py")
            is_javascript = server_script_path.endswith(".js")
            if not (is_python or is_javascript):
                raise ValueError("Server script must be a .py, .js file or npm package.")
        
            command = "python" if is_python else "node"
            
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        logger.debug(f"\n\nConnecting to stdio MCP server with command: {command} and args: {args}")

        # Start the server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.writer = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.writer))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"\n\nConnected to stdio MCP Server. Available tools: {[tool.name for tool in tools]}")

    async def connect_to_server(self, server_path_or_url: str):
        """Connect to an MCP server (either stdio or SSE).
        
        Args:
            server_path_or_url (str): Path to the server script or URL of SSE server.
        """
        # Check if the input is a URL (for SSE server)
        url_pattern = re.compile(r'^https?://')
        
        if url_pattern.match(server_path_or_url):
            # It's a URL, connect to SSE server
            await self.connect_to_sse_server(server_path_or_url)
        else:
            # It's a script path, connect to stdio server
            await self.connect_to_stdio_server(server_path_or_url)

    async def process_query(self, query: str, previous_messages: list = None) -> tuple[str, list]:
        """Process a query using the MCP server and available tools.
        
        Args:
            query (str): The query to send to the server.

        Returns:
            str: The response from the server.
        """
        model = "claude-3-5-sonnet-20241022"

        if not self.session:
            raise RuntimeError("Client session is not initialized.")
        
        messages = []
        if previous_messages:
            messages.extend(previous_messages)

        messages.append( 
            {
                "role": "user",
                "content": query
            }
        )
        logger.debug("Messages sent to LLM:", messages)
        
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": dict(tool.inputSchema) if tool.inputSchema else {}
             } for tool in response.tools]
        logger.debug("Available tools:", available_tools)

        # Initialize Claude API call
        logger.info(f"Sending query to {model}...")
        response = self.anthropic.messages.create(
            model=model,
            messages=messages,
            tools=available_tools,
            max_tokens=1000
        )

        # Process response and handle tool calls
        final_text = []
        assistant_message_content = []
        logger.debug("Response from LLM:", response.content)

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                logger.debug(f"Calling tool {tool_name} with args {tool_args}...")
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                logger.debug("Getting next response from LLM...")
                logger.debug("Messages sent to LLM:", messages)
                next_response = self.anthropic.messages.create(
                    model=model,
                    messages=messages,
                    tools=available_tools,
                    max_tokens=1000
                )
            
                logger.debug("Response from LLM:", next_response.content)
                final_text.append(next_response.content[0].text)
                messages.append({
                    "role": "assistant",
                    "content": next_response.content[0].text
                })


        return "\n".join(final_text), messages
    
    async def chat_loop(self):
        """Run an interactive chat loop with the server."""
        previous_messages = []
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                
                #  Check if the user wants to refresh conversation (history)
                if query.lower() == "refresh":
                    previous_messages = []
            
                response, previous_messages = await self.process_query(query, previous_messages=previous_messages)
                print("\nResponse:", response)
            except Exception as e:
                print("Error:", str(e))

    async def clenup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        if hasattr(self, '_session_context') and self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context') and self._streams_context:
            await self._streams_context.__aexit__(None, None, None)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_script_path_or_url>")
        print("Examples:")
        print("  - stdio server (npm): python client.py @playwright/mcp@latest")
        print("  - stdio server (python): python client.py ./weather.py")
        print("  - SSE server: python client.py http://localhost:3000/mcp")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.clenup()
        print("\nMCP Client Closed!")


if __name__ == "__main__":
    asyncio.run(main())
