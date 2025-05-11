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
        
        
        
        # Process tool calls if any
        if response_message.tool_calls:
            final_text.append(response_message.content or "")

            # Add the assistant's response to messages
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls
            })
            
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
        gemini_tools = self._convert_tools_to_gemini_format(available_tools)
        tools = genai_types.Tool(function_declarations=gemini_tools)
        config = genai_types.GenerateContentConfig(tools=[tools])
        
        # Prepare chat history
        chat_history = self._prepare_gemini_chat_history(previous_messages)
        
        chat = self.gemini.chats.create(
            model=model,
            config=config,
            history=chat_history
        )
        
        # Initialize variables for tracking conversation
        final_text = []
        messages = previous_messages.copy() if previous_messages else []
        messages.append({"role": "user", "content": query})
        
        try:
            logger.debug(f"Sending query to {model}...")
            response = chat.send_message(query)
            
            # Process the response
            final_text, messages = await self._process_gemini_response(
                response, 
                final_text, 
                messages, 
                model, 
                config
            )
                
        except Exception as e:
            logger.error(f"Error in Gemini processing: {str(e)}", exc_info=True)
            final_text.append(f"I encountered an error while processing your request: {str(e)}")
            
        return "\n".join(final_text), messages
    
    def _convert_tools_to_gemini_format(self, available_tools: list) -> list:
        """Convert tools from MCP format to Gemini format."""
        
        # Map JSON schema types to Gemini types
        type_mapping = {
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        gemini_tools = []
        for tool in available_tools:
            # Create basic tool structure
            function_declaration = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {"type": "OBJECT", "properties": {}, "required": []}
            }
            
            # Convert schema if available
            if "input_schema" in tool and tool["input_schema"]:
                schema = tool["input_schema"]
                
                # Add properties from the schema
                if "properties" in schema:
                    for prop_name, prop_details in schema["properties"].items():
                        prop_type = prop_details.get("type", "STRING").upper()
                        prop_type = type_mapping.get(prop_type.lower(), "STRING")
                            
                        property_schema = {"type": prop_type}
                        if "description" in prop_details:
                            property_schema["description"] = prop_details["description"]
                            
                        function_declaration["parameters"]["properties"][prop_name] = property_schema
                        
                # Add required properties
                if "required" in schema:
                    function_declaration["parameters"]["required"] = schema["required"]
                    
            gemini_tools.append(function_declaration)
        return gemini_tools
    
    def _prepare_gemini_chat_history(self, previous_messages: list) -> list:
        """Prepare chat history in Gemini's format."""
        chat_history = []
        if not previous_messages:
            return chat_history
            
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
        return chat_history
    
    async def _process_gemini_response(self, response, final_text, messages, model, config):
        """Process the response from Gemini, including any function calls."""
        if not hasattr(response, "candidates") or not response.candidates:
            logger.warning("No candidates in Gemini response")
            final_text.append("I couldn't generate a proper response.")
            return final_text, messages
            
        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not hasattr(candidate.content, "parts"):
            logger.warning("No content or parts in Gemini response")
            final_text.append("I received an incomplete response.")
            return final_text, messages
            
        # Process text and function calls
        for part in candidate.content.parts:
            # Process text part
            if hasattr(part, "text") and part.text:
                final_text.append(part.text)
                
            # Process function call part
            if hasattr(part, "function_call") and part.function_call:
                function_call = part.function_call
                tool_name = function_call.name
                
                # Parse tool arguments
                tool_args = self._parse_gemini_function_args(function_call)
                    
                # Add function call info to response
                function_call_text = f"I need to call the {tool_name} function to help with your request."
                final_text.append(function_call_text)
                
                # Execute tool call
                try:
                    logger.debug(f"Calling tool {tool_name} with args {tool_args}...")
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[tool results: {result}]")
                    
                    # Create a function response and send to Gemini for follow-up
                    final_text, messages = await self._handle_tool_result(
                        tool_name, 
                        function_call, 
                        result, 
                        final_text, 
                        messages,
                        model,
                        config
                    )
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    final_text.append(error_msg)
                
        return final_text, messages
    
    def _parse_gemini_function_args(self, function_call):
        """Parse function arguments from Gemini function call."""
        tool_args = {}
        try:
            if hasattr(function_call.args, "items"):
                for k, v in function_call.args.items():
                    tool_args[k] = v
            else:
                # Fallback if it's a string
                args_str = str(function_call.args)
                if args_str.strip():
                    tool_args = json.loads(args_str)
                    
            logger.debug(f"Parsed tool args: {tool_args}")
        except Exception as e:
            logger.error(f"Failed to parse function args: {e} - {type(function_call.args)}", exc_info=True)
            
        return tool_args
    
    async def _handle_tool_result(self, tool_name, function_call, result, final_text, messages, model, config):
        """Handle the result of a tool call and get follow-up response."""
        try:
            # Prepare function response
            function_response_part = genai_types.Part.from_function_response(
                name=tool_name,
                response={"result": result.content if hasattr(result, "content") else str(result)},
            )

            # Prepare contents for follow-up
            contents = [
                genai_types.Content(
                    role="model", 
                    parts=[genai_types.Part(function_call=function_call)]
                )
            ]
            
            # Add to messages history                               
            messages.append({
                "role": "assistant", 
                "content": function_call.model_dump_json()
            })

            # Add function response to contents
            contents.append(
                genai_types.Content(
                    role="user", 
                    parts=[function_response_part]
                )
            )
            
            # Add to messages history
            result_content = result.content if hasattr(result, "content") else str(result)                           
            messages.append({
                "role": "user", 
                "content": {"result": result_content}
            })
           
            # Send function response to get final answer
            follow_up_response = self.gemini.models.generate_content(
                model=model,
                config=config,
                contents=contents,
            )
            
            # Extract text from follow-up response
            if hasattr(follow_up_response, "candidates") and follow_up_response.candidates:
                follow_up_candidate = follow_up_response.candidates[0]
                if (hasattr(follow_up_candidate, "content") and 
                    hasattr(follow_up_candidate.content, "parts")):
                    
                    follow_up_text = ""
                    for follow_up_part in follow_up_candidate.content.parts:
                        if hasattr(follow_up_part, "text"):
                            follow_up_text += follow_up_part.text
                            
                    if follow_up_text:
                        final_text.append(follow_up_text)
                        messages.append({
                            "role": "assistant", 
                            "content": follow_up_text
                        })
                    else:
                        final_text.append("I received the tool results but couldn't generate a follow-up response.")
                        
            else:
                final_text.append("I processed your request but couldn't generate a follow-up response.")
                
        except Exception as e:
            logger.error(f"Error in follow-up response: {str(e)}", exc_info=True)
            final_text.append(f"I received the tool results but encountered an error: {str(e)}")
            
        return final_text, messages

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
