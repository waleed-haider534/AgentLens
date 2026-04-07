# AgentLens

AI-powered LLM discovery assistant for agentic AI workflows — built with OpenAI Responses API, Ollama, and Streamlit

## Overview

AgentLens helps developers discover the best Large Language Models (LLMs) for their agentic AI workflows. Simply describe your use case, and AgentLens analyzes your requirements to recommend suitable LLMs with detailed comparisons including parameters, providers, features, tool calling support, and cost tiers.

## Features

- **Smart LLM Recommendations**: Describe your workflow and get AI-powered recommendations
- **Local & Cloud Models**: Compare local Ollama models with cloud alternatives
- **Interactive Testing**: Test local Ollama models directly from the interface
- **Side-by-Side Comparison**: Visual comparison table of recommended models
- **Search History**: Track your recent searches for quick reference

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLM Engine**: Ollama (local inference)
- **Configuration**: python-dotenv

## Prerequisites

- Python 3.10+
- Ollama installed and running locally

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AgentLens
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
# Edit .env file with your settings
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

5. Ensure Ollama is running:
```bash
ollama serve
```

## Usage

Start the application:
```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`.

### How to Use

1. **Describe your workflow** in the text area (e.g., "I'm building a marketing automation agent that handles campaign creation, audience targeting, and report generation")
2. Click **Search LLMs** to get recommendations
3. Review the recommended models with detailed information
4. Use the **Comparison Table** to compare models side-by-side
5. **Test local models** directly from the interface

## Project Structure

```
AgentLens/
├── app.py              # Streamlit frontend application
├── agent_core.py       # Core LLM recommendation logic
├── config.py           # Configuration and system prompts
├── ollama_utils.py     # Ollama API utilities
├── main.py            # Entry point
├── requirements.txt   # Python dependencies
├── .env               # Environment configuration
└── README.md          # This file
```

## Configuration

Edit the `.env` file to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Model for analyzing queries | `llama3.2` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |

## How It Works

1. User describes their agentic workflow use case
2. AgentLens sends the query to a local Ollama model
3. The model analyzes requirements and recommends suitable LLMs
4. Results are displayed as interactive cards with detailed information
5. Users can compare models and test local models directly
