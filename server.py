"""
LitServe Server with CrewAI Orchestration
Uses Ollama with Qwen 3 for local LLM inference
"""
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from crewai.llm import LLM
from pydantic import BaseModel, Field
from litserve import LitAPI, LitServer

load_dotenv()


# ============================================================
# Ollama LLM Configuration
# ============================================================

def get_ollama_llm():
    """Get the Ollama LLM instance with Qwen 3."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3")
    
    return LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        # Additional Ollama parameters
        temperature=0.7,
        max_tokens=2000,
    )


# ============================================================
# Vector Database Tool
# ============================================================

class VectorDBTool(BaseTool):
    """Tool for retrieving context from a vector database."""
    
    name: str = "Vector Database Retrieval"
    description: str = "Retrieves relevant context from the vector database based on the user query."
    
    def __init__(self, index: Optional[faiss.Index] = None, 
                 texts: Optional[List[str]] = None,
                 embedding_model: Optional[SentenceTransformer] = None,
                **kwargs):
        super().__init__(**kwargs)
        self._index = index
        self._texts = texts or []
        self._embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        
        if self._index is None and self._texts:
            self._create_index()
    
    def _create_index(self):
        """Create FAISS index from texts."""
        embeddings = self._embedding_model.encode(self._texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings)
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant documents from vector DB."""
        if self._index is None:
            return "Vector database is empty. No context available."
        
        query_embedding = self._embedding_model.encode([query])
        distances, indices = self._index.search(query_embedding, min(top_k, len(self._texts)))
        
        results = []
        for idx in indices[0]:
            if idx < len(self._texts):
                results.append(self._texts[idx])
        
        if not results:
            return "No relevant context found in vector database."
        
        return "\n\n".join(results)


# ============================================================
# Firecrawl Web Search Tool
# ============================================================

class FirecrawlWebSearchTool(BaseTool):
    """Tool for web search using Firecrawl."""
    
    name: str = "Web Search"
    description: str = "Searches the internet for relevant information using Firecrawl. Use this when you need up-to-date information or context not available in the vector database."
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
    
    def _run(self, query: str, limit: int = 5) -> str:
        """Search the web for information."""
        try:
            from firecrawl import FirecrawlApp
            
            if not self._api_key:
                return "Error: Firecrawl API key not configured. Please set FIRECRAWL_API_KEY environment variable."
            
            app = FirecrawlApp(api_key=self._api_key)
            result = app.search(query=query, limit=limit)
            
            if not result or not result.get('data'):
                return f"No web results found for: {query}"
            
            formatted_results = []
            for item in result.get('data', [])[:limit]:
                title = item.get('title', 'No title')
                content = item.get('content', item.get('description', 'No content'))
                url = item.get('url', '')
                formatted_results.append(f"## {title}\n{content}\nSource: {url}")
            
            return "\n\n".join(formatted_results)
            
        except ImportError:
            return "Error: Firecrawl package not installed. Run: pip install firecrawl-sdk"
        except Exception as e:
            return f"Error during web search: {str(e)}"


# ============================================================
# Sample Vector Database
# ============================================================

def create_sample_vector_db() -> VectorDBTool:
    """Create a sample vector database with example documents."""
    
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.",
        "Large Language Models (LLMs) are AI models trained on vast amounts of text data.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.",
        "Vector databases store embeddings and enable efficient similarity search.",
        "CrewAI is a framework for building multi-agent AI systems with role-based agents.",
        "Firecrawl is a tool for scraping and searching web content programmatically.",
        "LitServe is a deployment framework for AI models by LightningAI.",
        "Agentic RAG systems use AI agents to dynamically choose retrieval sources.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "Qwen 3 is a powerful open-source language model developed by Alibaba.",
        "Ollama allows running large language models locally on your machine."
    ]
    
    return VectorDBTool(texts=sample_documents)


# ============================================================
# Research Agent and Task Definition
# ============================================================

def create_research_agent_and_task(
    vector_db_tool: VectorDBTool,
    web_search_tool: FirecrawlWebSearchTool,
    llm: LLM
) -> tuple[Agent, Task]:
    """
    Create the Research Agent and Task.
    
    This Agent accepts the user query and retrieves the relevant context
    using the vectorDB tool and a web search tool powered by Firecrawl.
    
    Args:
        vector_db_tool: Tool for retrieving context from vector database
        web_search_tool: Tool for web search using Firecrawl
        llm: The language model to use
    
    Returns:
        Tuple of (Agent, Task)
    """
    # Define the tools available to the Research Agent
    research_tools = [vector_db_tool, web_search_tool]
    
    # Create the Research Agent
    research_agent = Agent(
        role="Research Agent",
        goal="Thoroughly research the user query by retrieving context from both vector database and web search",
        backstory=(
            "You are a research specialist who combines multiple sources of information. "
            "You use both the vector database (for internal knowledge) and web search (for up-to-date information) "
            "to provide comprehensive answers. Your job is to accept a user query and retrieve "
            "relevant context using both the vectorDB tool and Firecrawl web search tool."
        ),
        tools=research_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Create the Research Task
    research_task = Task(
        description=(
            "Research the given query by using BOTH retrieval tools:\n"
            "1. Use the Vector Database Retrieval tool to search for internal knowledge.\n"
            "2. Use the Web Search tool to find up-to-date information from the internet.\n"
            "3. Combine all results into a comprehensive research report.\n\n"
            "User Query: {query}\n\n"
            "Always cite sources when possible."
        ),
        expected_output="A comprehensive research report combining internal knowledge from vector DB and up-to-date web sources",
        agent=research_agent
    )
    
    return research_agent, research_task


# ============================================================
# Crew Orchestration Setup
# ============================================================

class AgenticRAGCrew:
    """
    Crew orchestration class that sets up all agents and tasks.
    This is called from the LitServe setup() method.
    """
    
    def __init__(self, vector_db_tool: VectorDBTool, web_search_tool: FirecrawlWebSearchTool):
        self.vector_db_tool = vector_db_tool
        self.web_search_tool = web_search_tool
        self.tools = [self.vector_db_tool, self.web_search_tool]
        
        # Initialize Ollama LLM
        self.llm = get_ollama_llm()
        
        # Create agents
        self.retriever_agent = self._create_retriever_agent()
        self.writer_agent = self._create_writer_agent()
        
        # Create Research Agent and Task explicitly
        self.research_agent, self.research_task = create_research_agent_and_task(
            vector_db_tool=vector_db_tool,
            web_search_tool=web_search_tool,
            llm=self.llm
        )
    
    def _create_retriever_agent(self) -> Agent:
        """Create the Retriever Agent."""
        return Agent(
            role="Retriever Agent",
            goal="Retrieve the most relevant context for the user query by choosing the best available tool",
            backstory=(
                "You are an expert at information retrieval. Your job is to analyze user queries "
                "and retrieve relevant context using either the vector database or web search. "
                "You choose the most appropriate tool based on the query type."
            ),
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def _create_writer_agent(self) -> Agent:
        """Create the Writer Agent."""
        return Agent(
            role="Writer Agent",
            goal="Generate a clear, accurate, and helpful response based on the retrieved context",
            backstory=(
                "You are an expert writer specializing in technical explanations. "
                "You take the retrieved context and craft a comprehensive, easy-to-understand response. "
                "Always cite your sources when possible."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def run_rag(self, query: str) -> str:
        """Run the standard RAG pipeline."""
        
        retriever_task = Task(
            description=f"Retrieve relevant context for the following query: {query}",
            expected_output="Relevant context from the best available source (vector DB or web search)",
            agent=self.retriever_agent
        )
        
        writer_task = Task(
            description="Generate a final response based on the retrieved context",
            expected_output="A clear, accurate, and well-structured response",
            agent=self.writer_agent,
            context=[retriever_task]
        )
        
        crew = Crew(
            agents=[self.retriever_agent, self.writer_agent],
            tasks=[retriever_task, writer_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    def run_research(self, query: str) -> str:
        """Run the research pipeline using both retrieval methods."""
        
        # Create a task instance with the actual query
        research_task = Task(
            description=(
                f"Research the following query thoroughly by using both the vector database "
                f"and web search tools: {query}\n\n"
                f"1. First, search the vector database for internal knowledge.\n"
                f"2. Then, search the web for up-to-date information.\n"
                f"3. Combine the results into a comprehensive answer."
            ),
            expected_output="A comprehensive research report combining internal knowledge and web sources",
            agent=self.research_agent
        )
        
        crew = Crew(
            agents=[self.research_agent],
            tasks=[research_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)


# ============================================================
# LitServe API
# ============================================================

class RAGLitAPI(LitAPI):
    """LitServe API for the Agentic RAG Pipeline."""
    
    def __init__(self):
        self.crew = None
        self.vector_db_tool = None
        self.web_search_tool = None
    
    def setup(self, device: str = "cuda"):
        """
        Setup method - initializes the CrewAI orchestration with all agents and tools.
        This is the main entry point for initializing the pipeline.
        
        The setup includes:
        1. Vector Database Tool (FAISS-based retrieval)
        2. Firecrawl Web Search Tool
        3. Retriever Agent - chooses best tool for retrieval
        4. Writer Agent - generates final response
        5. Research Agent - retrieves from both vector DB and web search
        
        The Research Agent and Task are explicitly defined to accept user queries
        and retrieve context using the vectorDB tool and Firecrawl web search tool.
        """
        print(f"\n{'='*60}")
        print("Initializing Agentic RAG Pipeline with CrewAI...")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
        
        # Initialize tools
        print("📦 Creating vector database tool...")
        self.vector_db_tool = create_sample_vector_db()
        
        print("🌐 Creating web search tool (Firecrawl)...")
        self.web_search_tool = FirecrawlWebSearchTool()
        
        # Create Crew orchestration with all agents
        print("🤖 Setting up CrewAI agents...")
        print("   - Retriever Agent (chooses vector DB or web search)")
        print("   - Writer Agent (generates response)")
        print("   - Research Agent (uses both vector DB + web search)")
        
        self.crew = AgenticRAGCrew(
            vector_db_tool=self.vector_db_tool,
            web_search_tool=self.web_search_tool
        )
        
        # Explicitly show Research Agent and Task info
        print(f"\n📋 Research Agent Configuration:")
        print(f"   Role: {self.crew.research_agent.role}")
        print(f"   Goal: {self.crew.research_agent.goal}")
        print(f"   Tools: {[t.name for t in self.crew.research_agent.tools]}")
        print(f"   Task: {self.crew.research_task.description[:100]}...")
        
        print(f"\n{'='*60}")
        print("✅ Setup complete!")
        print(f"   LLM: Ollama ({os.getenv('OLLAMA_MODEL', 'qwen3')})")
        print(f"   Tools: {[tool.name for tool in self.crew.tools]}")
        print(f"{'='*60}\n")
    
    def decode_request(self, request: Any) -> Dict:
        """Decode the incoming request."""
        return request
    
    def predict(self, inputs: Dict) -> Dict:
        """Run prediction on the inputs."""
        query = inputs.get("query", "")
        mode = inputs.get("mode", "research")
        
        if not query:
            return {"error": "Query is required"}
        
        try:
            if mode == "research":
                result = self.crew.run_research(query)
            else:
                result = self.crew.run_rag(query)
            
            return {
                "query": query,
                "mode": mode,
                "response": result
            }
        except Exception as e:
            return {
                "query": query,
                "mode": mode,
                "error": str(e)
            }
    
    def encode_response(self, output: Dict) -> Any:
        """Encode the response."""
        return output


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function to start the LitServe server."""
    
    print("\n" + "="*60)
    print("🚀 Starting Agentic RAG Server")
    print("="*60)
    
    # Create and start the server
    api = RAGLitAPI()
    server = LitServer(api, debug=True)
    
    print("\n📡 Server ready at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("\nExample request:")
    print('  curl -X POST http://localhost:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What is RAG?", "mode": "research"}\'')
    print("\n" + "="*60 + "\n")
    
    server.run(port=8000)


if __name__ == "__main__":
    main()
