# MCP Client

A Python client for connecting to Machine-to-Machine Communication Protocol (MCP) servers. This client allows you to interact with both stdio and SSE MCP servers, enabling LLM agents to use tools through a standardized protocol.

## Features

- Connect to stdio MCP servers (Python and JavaScript)
- Connect to SSE MCP servers
- Interactive chat interface with multiple LLM options:
  - Anthropic Claude 3.5 Sonnet
  - OpenAI GPT-4o
  - Google gemini-2.0-flash
- Support for tool calling and result processing
- Conversation history management

## Prerequisites

- Python 3.8+
- API keys set as environment variables:
  - Anthropic API key
  - OpenAI API key
  - Google API key (for Gemini)

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install mcp-protocol-client anthropic openai aiohttp python-dotenv google-genai
```

3. Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

4. Create a `logs` directory to store client logs:

```bash
mkdir logs
```

## Usage

### Basic Usage

```bash
python client.py <server_script_path_or_url> [llm_provider]
```

The `llm_provider` parameter can be either `anthropic` (default), `openai`, or `gemini`.

### Examples

1. Connect to a stdio server (npm package) using Anthropic:

```bash
python client.py @playwright/mcp@latest
```

2. Connect to a stdio server (Python script) using OpenAI:

```bash
python client.py ./weather.py openai
```

3. Connect to an SSE server using Gemini:

```bash
python client.py http://localhost:3000/mcp gemini
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
   - Sending the query to the chosen LLM (Claude, GPT, or Gemini)
   - Detecting tool calls in the response
   - Executing tool calls on the MCP server
   - Passing tool results back to the LLM for final response generation
4. Maintains conversation history for context

## Logging

Logs are stored in `logs/mcp_client.log` and are also displayed in the console.
