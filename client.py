import asyncio
import sys
import logging
import json
import re

from typing import Optional, Literal
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from anthropic import Anthropic
from openai import AsyncOpenAI
from google import genai
from google.genai import types as genai_types

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
    def __init__(self, llm_provider: Literal["anthropic", "openai", "gemini"] = "anthropic"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_provider = llm_provider
        self.anthropic = None
        self.openai = None
        self.gemini = None
        
        # Initialize the selected LLM client
        if llm_provider == "anthropic":
            self.anthropic = Anthropic()
        elif llm_provider == "openai":
            self.openai = AsyncOpenAI()
        elif llm_provider == "gemini":
            self.gemini = genai.Client()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'anthropic', 'openai', or 'gemini'.")
        
        logger.info(f"Initialized MCPClient with {llm_provider} as the LLM provider")

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
            previous_messages (list, optional): Previous conversation history.

        Returns:
            tuple[str, list]: The response from the server and updated messages.
        """
        if not self.session:
            raise RuntimeError("Client session is not initialized.")
        
        # Get available tools
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": dict(tool.inputSchema) if tool.inputSchema else {}
        } for tool in response.tools]
        
        if self.llm_provider == "anthropic":
            return await self._process_query_anthropic(query, available_tools, previous_messages)
        elif self.llm_provider == "openai":
            return await self._process_query_openai(query, available_tools, previous_messages)
        elif self.llm_provider == "gemini":
            return await self._process_query_gemini(query, available_tools, previous_messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    async def _process_query_anthropic(self, query: str, available_tools: list, previous_messages: list = None) -> tuple[str, list]:
        """Process a query using Anthropic's Claude."""
        model = "claude-3-5-sonnet-20241022"
        
        messages = []
        if previous_messages:
            messages.extend(previous_messages)

        messages.append( 
            {
                "role": "user",
                "content": query
            }
        )
        logger.debug("Messages sent to Claude:", messages)
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
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[tool results: {result}]")
                
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
                logger.debug("Getting next response from Claude...")
                logger.debug("Messages sent to Claude:", messages)
                next_response = self.anthropic.messages.create(
                    model=model,
                    messages=messages,
                    tools=available_tools,
                    max_tokens=1000
                )
            
                logger.debug("Response from Claude:", next_response.content)
                final_text.append(next_response.content[0].text)
                messages.append({
                    "role": "assistant",
                    "content": next_response.content[0].text
                })

        return "\n".join(final_text), messages
    
    async def _process_query_openai(self, query: str, available_tools: list, previous_messages: list = None) -> tuple[str, list]:
        """Process a query using OpenAI's GPT models."""
        model = "gpt-4o"
        
        # Convert available_tools to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            } for tool in available_tools
        ]
        
        messages = []
        if previous_messages:
            messages.extend(previous_messages)
            
        messages.append({
            "role": "user",
            "content": query
        })
        
        logger.debug("Messages sent to OpenAI:", messages)
        logger.debug("Available tools:", openai_tools)
        
        # Initialize OpenAI API call
        logger.info(f"Sending query to {model}...")
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        final_text = []
        
        # Add the assistant's response to messages
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": response_message.tool_calls
        })
        
        # Process tool calls if any
        if response_message.tool_calls:
            final_text.append(response_message.content or "")
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute tool call
                logger.debug(f"Calling tool {function_name} with args {function_args}...")
                final_text.append(f"[Calling tool {function_name} with args {function_args}]")
                result = await self.session.call_tool(function_name, function_args)
                final_text.append(f"[tool results: {result}]")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content
                })
            
            # Get the final response after tool calls
            logger.debug("Getting next response from OpenAI...")
            next_response = await self.openai.chat.completions.create(
                model=model,
                messages=messages
            )
            
            logger.debug("Response from OpenAI:", next_response.choices[0].message)
            final_text.append(next_response.choices[0].message.content)
            messages.append({
                "role": "assistant", 
                "content": next_response.choices[0].message.content
            })
        else:
            final_text.append(response_message.content)
        
        return "\n".join(final_text), messages
    
    async def _process_query_gemini(self, query: str, available_tools: list, previous_messages: list = None) -> tuple[str, list]:
        """Process a query using Google's Gemini models."""
        model = "gemini-2.0-flash"
        
        # Convert available_tools to a format suitable for Gemini
        gemini_tools = []
        for tool in available_tools:
            # Gemini needs a specific schema format - convert from JSON Schema to Gemini's expected format
            parameters = {}
            if "input_schema" in tool and tool["input_schema"]:
                schema = tool["input_schema"]
                
                # Start with basic structure
                parameters = {
                    "type": "OBJECT",
                    "properties": {},
                    "required": []
                }
                
                # Add properties from the schema
                if "properties" in schema:
                    for prop_name, prop_details in schema["properties"].items():
                        prop_type = prop_details.get("type", "STRING").upper()
                        # Convert JSON schema types to Gemini types
                        if prop_type.lower() == "number":
                            prop_type = "NUMBER"
                        elif prop_type.lower() == "integer":
                            prop_type = "INTEGER"
                        elif prop_type.lower() == "boolean":
                            prop_type = "BOOLEAN"
                        elif prop_type.lower() == "array":
                            prop_type = "ARRAY"
                        elif prop_type.lower() == "object":
                            prop_type = "OBJECT"
                        else:
                            prop_type = "STRING"
                            
                        property_schema = {"type": prop_type}
                        if "description" in prop_details:
                            property_schema["description"] = prop_details["description"]
                            
                        parameters["properties"][prop_name] = property_schema
                        
                # Add required properties
                if "required" in schema:
                    parameters["required"] = schema["required"]
            
            function_declaration = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": parameters
            }
            gemini_tools.append(function_declaration)

        tools = genai_types.Tool(function_declarations=gemini_tools)
        config = genai_types.GenerateContentConfig(tools=[tools])
        model_instance = self.gemini
        
        # Prepare chat history
        # Add previous messages to chat history if provided
        chat_history = []
        if previous_messages:
            for message in previous_messages:
                if message["role"] == "user" and isinstance(message["content"], str):
                    chat_history.append({
                        "role": "user",
                        "parts": [{"text": message["content"]}]
                    })
                elif message["role"] == "assistant" and isinstance(message["content"], str):
                    chat_history.append({
                        "role": "model",
                        "parts": [{"text": message["content"]}]
                    })

        chat = model_instance.chats.create(
            model=model,
            config=config,
            history=chat_history
        )
        
        final_text = []
        messages = previous_messages.copy() if previous_messages else []
        # Add current query to messages
        messages.append({"role": "user", "content": query})
        
        # Send the request to Gemini
        try:
            logger.debug(f"Sending query to {model}...")
            response = chat.send_message(query)
            
            # Handle function calls first because response.text may fail if there's a function call
            final_text = []
            has_function_call = False
            if hasattr(response, "candidates") and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        # Check if part is text
                        if hasattr(part, "text") and part.text:
                            final_text.append(part.text)
                            
                        # Check if part is a function call
                        if hasattr(part, "function_call"):
                            function_call = part.function_call

                            if function_call:
                                has_function_call = True
                                tool_name = function_call.name
                                
                                try:
                                    if hasattr(function_call.args, "items"):
                                        tool_args = {}
                                        for k, v in function_call.args.items():
                                            tool_args[k] = v
                                    else:
                                        # Fallback if it's a string (which is rare but possible)
                                        tool_args = json.loads(str(function_call.args))
                                        
                                    logger.debug(f"Parsed tool args: {tool_args}")
                                except Exception as e:
                                    logger.error(f"Failed to parse function args: {e} - {type(function_call.args)}")
                                    tool_args = {}
                                    
                                # Show what function was called
                                function_call_text = f"I need to call the {tool_name} function to help with your request."
                                final_text.append(function_call_text)
                                
                                # Execute tool call
                                logger.debug(f"Calling tool {tool_name} with args {tool_args}...")
                                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                                result = await self.session.call_tool(tool_name, tool_args)
                                final_text.append(f"[tool results: {result}]")
                                
                                # Create a function response part
                                function_response_part = genai_types.Part.from_function_response(
                                    name=tool_name,
                                    response={"result": result},
                                )

                                # Append function call and result of the function execution response to contents
                                contents = [
                                    genai_types.Content(role="model", parts=[genai_types.Part(function_call=function_call)]) 
                                ]

                                contents.append(
                                    genai_types.Content(role="user", parts=[function_response_part])
                                )

                                # Send function response to get final answer
                                try:
                                    follow_up_response = model_instance.models.generate_content(
                                        model=model,
                                        config=config,
                                        contents=contents,
                                    )
                                    
                                    # Get text from follow-up response parts
                                    if hasattr(follow_up_response, "candidates") and len(follow_up_response.candidates) > 0:
                                        follow_up_candidate = follow_up_response.candidates[0]
                                        if hasattr(follow_up_candidate, "content") and hasattr(follow_up_candidate.content, "parts"):
                                            follow_up_text = ""
                                            for follow_up_part in follow_up_candidate.content.parts:
                                                if hasattr(follow_up_part, "text"):
                                                    follow_up_text += follow_up_part.text
                                                    
                                            final_text.append(follow_up_text)
                                    
                                except Exception as e:
                                    logger.error(f"Error in follow-up response: {str(e)}")
                                    final_text.append("Error processing function result.")
            
            # If we didn't find any function calls, try to get the full response text
            if not has_function_call and not final_text:
                try:
                    assistant_response_text = response.text
                    final_text.append(assistant_response_text)
                except ValueError as e:
                    logger.error(f"Error extracting text from response: {str(e)}")
                    # Try to extract text directly from parts
                    if hasattr(response, "candidates") and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    final_text.append(part.text)
            
            # Construct a simplified text representation for the message history
            simplified_response = " ".join(final_text)
            messages.append({"role": "assistant", "content": simplified_response})
            
        except Exception as e:
            logger.error(f"Error in Gemini processing: {str(e)}")
            raise
            
        return "\n".join(final_text), messages
    
    async def chat_loop(self):
        """Run an interactive chat loop with the server."""
        previous_messages = []
        print("Type your queries or 'quit' to exit.")
        print("Type 'refresh' to clear conversation history.")
        print(f"Using {self.llm_provider.upper()} as the LLM provider.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                
                #  Check if the user wants to refresh conversation (history)
                if query.lower() == "refresh":
                    previous_messages = []
                    print("Conversation history cleared.")
                    continue
            
                response, previous_messages = await self.process_query(query, previous_messages=previous_messages)
                print("\nResponse:", response)
            except Exception as e:
                logger.exception("Error in chat loop")
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
        print("Usage: python client.py <server_script_path_or_url> [llm_provider]")
        print("Examples:")
        print("  - stdio server (npm): python client.py @playwright/mcp@latest")
        print("  - stdio server (python): python client.py ./weather.py")
        print("  - SSE server: python client.py http://localhost:3000/mcp")
        print("  - Specify LLM provider: python client.py ./weather.py openai")
        print("  - Use Gemini: python client.py ./weather.py gemini")
        sys.exit(1)

    # Default to anthropic if provider isn't specified
    llm_provider = "anthropic"
    if len(sys.argv) > 2 and sys.argv[2].lower() in ["anthropic", "openai", "gemini"]:
        llm_provider = sys.argv[2].lower()

    client = MCPClient(llm_provider=llm_provider)
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.clenup()
        print("\nMCP Client Closed!")


if __name__ == "__main__":
    asyncio.run(main())
