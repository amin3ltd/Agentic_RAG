# Agentic RAG Pipeline

A powerful Retrieval-Augmented Generation (RAG) pipeline with agentic capabilities that dynamically fetches context from multiple sources.

## Features

- **Multi-Source Retrieval**: Dynamically chooses between vector database and web search
- **CrewAI Orchestration**: Multi-agent system with Retriever, Writer, and Research agents
- **Firecrawl Integration**: Web search capabilities for up-to-date information
- **LitServe Deployment**: Production-ready API server using LightningAI's LitServe
- **Local LLM Support**: Uses Ollama with Qwen 3 for privacy and offline capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Retriever Agent                            │
│         (Chooses best tool: Vector DB or Web Search)        │
└───────┬─────────────────────────────────────┬───────────────┘
        │                                     │
        ▼                                     ▼
┌───────────────┐                   ┌─────────────────────┐
│ Vector DB     │                   │ Firecrawl Web      │
│ (FAISS)       │                   │ Search              │
└───────┬───────┘                   └──────────┬──────────┘
        │                                      │
        └──────────────────┬───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Writer Agent                              │
│              (Generates final response)                      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amin3ltd/Agentic_RAG.git
cd Agentic_RAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys (optional - uses local Ollama by default)
```

5. Install and start Ollama (for local LLM):
```bash
# Install Ollama: https://ollama.ai/
ollama pull qwen3
ollama serve
```

## Usage

### Running the Server

```bash
python server.py
```

The server will start on `http://localhost:8000`

### API Endpoints

- `POST /predict` - Run the RAG pipeline
- `GET /health` - Health check

### Example Request

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "query": "What is Retrieval-Augmented Generation?",
    "mode": "research"  # or "rag"
})

print(response.json())
```

### Example cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG in AI?", "mode": "research"}'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name to use | `qwen3` |
| `FIRECRAWL_API_KEY` | Firecrawl API key (optional) | - |
| `OPENAI_API_KEY` | OpenAI API key (fallback) | - |

### Vector Database

The default vector database is populated with sample documents about:
- Machine Learning
- Natural Language Processing
- Large Language Models
- RAG
- CrewAI
- Firecrawl
- LitServe

To use your own documents, modify the `create_sample_vector_db()` function in `app.py`.

## Project Structure

```
Agentic_RAG/
├── app.py              # Core RAG pipeline and tools
├── server.py           # LitServe server with Crew orchestration
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore patterns
└── README.md           # This file
```

## Agents

### Retriever Agent
- **Role**: Information Retrieval Specialist
- **Goal**: Retrieve relevant context by choosing the best available tool
- **Tools**: Vector DB, Web Search

### Writer Agent
- **Role**: Technical Writer
- **Goal**: Generate clear, accurate responses from retrieved context

### Research Agent
- **Role**: Research Specialist
- **Goal**: Combine multiple sources for comprehensive answers
- **Tools**: Vector DB, Web Search

## Tech Stack

- **CrewAI**: Multi-agent orchestration
- **Firecrawl**: Web search and scraping
- **LitServe**: Model serving framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **Ollama**: Local LLM runtime (Qwen 3)

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
