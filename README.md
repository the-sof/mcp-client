# MCP Client

A Python client for connecting to Machine-to-Machine Communication Protocol (MCP) servers. This client allows you to interact with both stdio and SSE MCP servers, enabling LLM agents to use tools through a standardized protocol.

## Features

- Connect to stdio MCP servers (Python and JavaScript)
- Connect to SSE MCP servers
- Interactive chat interface with multiple LLM options:
  - Anthropic Claude 3.5 Sonnet (default)
  - OpenAI GPT-4o
  - Google Gemini 2.0 Flash
- Tool calling with automatic result processing
- Conversation history management with refresh capability
- Detailed logging for debugging and monitoring

## Prerequisites

- Python 3.8+
- API keys set as environment variables:
  - `ANTHROPIC_API_KEY` for Anthropic Claude
  - `OPENAI_API_KEY` for OpenAI GPT models
  - `GOOGLE_API_KEY` for Google Gemini

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install mcp-protocol-client anthropic openai google-genai python-dotenv
```

3. Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
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

Where:
- `<server_script_path_or_url>` is either a path to an MCP server script or URL to an SSE MCP server
- `[llm_provider]` is optional and can be one of: `anthropic` (default), `openai`, or `gemini`

### Examples

1. Connect to a Python MCP server using Anthropic Claude (default):

```bash
python client.py ./weather.py
```

2. Connect to a JavaScript npm MCP server using OpenAI:

```bash
python client.py @playwright/mcp@latest openai
```

3. Connect to an SSE MCP server using Google Gemini:

```bash
python client.py http://localhost:8000/sse gemini
```

### Interactive Chat Commands

- Type your queries to interact with the LLM and tools
- Type `refresh` to clear conversation history
- Type `quit` to exit the application

## Development

The client includes VS Code launch configurations for various setups, making it easy to debug and test with different servers and LLM providers.

## How It Works

The MCP Client:
1. Connects to an MCP server (either stdio or SSE)
2. Lists available tools from the server
3. Processes user queries by:
   - Sending the query to the selected LLM with available tools
   - Detecting and executing tool calls when the LLM requests them
   - Sending tool results back to the LLM for processing
   - Providing the final response to the user
4. Maintains conversation history for context

## Logging

Logs are stored in `logs/mcp_client.log` and are also displayed in the console. The logging level can be adjusted in the client.py file.
