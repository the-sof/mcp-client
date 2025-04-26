# MCP Client

A Python client for connecting to Machine-to-Machine Communication Protocol (MCP) servers. This client allows you to interact with both stdio and SSE MCP servers, enabling LLM agents to use tools through a standardized protocol.

## Features

- Connect to stdio MCP servers (Python and JavaScript)
- Connect to SSE MCP servers
- Interactive chat interface with Claude 3.5 Sonnet
- Support for tool calling and result processing
- Conversation history management

## Prerequisites

- Python 3.8+
- Anthropic API key set as environment variable

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install mcp-protocol-client anthropic aiohttp python-dotenv
```

3. Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

4. Create a `logs` directory to store client logs:

```bash
mkdir logs
```

## Usage

### Basic Usage

```bash
python client.py <server_script_path_or_url>
```

### Examples

1. Connect to a stdio server (npm package):

```bash
python client.py @playwright/mcp@latest
```

2. Connect to a stdio server (Python script):

```bash
python client.py ./weather.py
```

3. Connect to an SSE server:

```bash
python client.py http://localhost:3000/mcp
```

### Interactive Chat Commands

- Type your queries to interact with the LLM and tools
- Type `refresh` to clear conversation history
- Type `quit` to exit the application

## How It Works

The MCP Client:
1. Connects to an MCP server (either stdio or SSE)
2. Lists the available tools from the server
3. Processes user queries by:
   - Sending the query to Claude 3.5 Sonnet
   - Detecting tool calls in the response
   - Executing tool calls on the MCP server
   - Passing tool results back to Claude for final response generation
4. Maintains conversation history for context

## Logging

Logs are stored in `logs/mcp_client.log` and are also displayed in the console.
