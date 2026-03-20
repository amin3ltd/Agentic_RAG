"""
Agentic RAG Pipeline with CrewAI, Firecrawl, and LitServe
"""
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from crewai import Agent, Task, Crew, Tool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from litserve import LitAPI, LitServer

load_dotenv()


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
        
        # Create index if not provided
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
        
        # Embed query
        query_embedding = self._embedding_model.encode([query])
        
        # Search
        distances, indices = self._index.search(query_embedding, min(top_k, len(self._texts)))
        
        # Get relevant texts
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
            
            # Search for the query
            result = app.search(query=query, limit=limit)
            
            if not result or not result.get('data'):
                return f"No web results found for: {query}"
            
            # Format results
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
# Create Sample Vector Database
# ============================================================

def create_sample_vector_db() -> VectorDBTool:
    """Create a sample vector database with example documents."""
    
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.",
        "Large Language Models (LLMs) are AI models trained on vast amounts of text data.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.",
        "Vector databases store embeddings and enable efficient similarity search.",
        "CrewAI is a framework for building multi-agent AI systems.",
        "Firecrawl is a tool for scraping and searching web content.",
        "LitServe is a deployment framework for AI models by LightningAI.",
        "Agentic RAG systems use AI agents to dynamically choose retrieval sources.",
        "Embeddings are numerical representations of text that capture semantic meaning."
    ]
    
    return VectorDBTool(texts=sample_documents)


# ============================================================
# CrewAI Agents Setup
# ============================================================

class AgenticRAGPipeline:
    """Main RAG pipeline with CrewAI agents."""
    
    def __init__(self, vector_db_tool: VectorDBTool, web_search_tool: FirecrawlWebSearchTool):
        self.vector_db_tool = vector_db_tool
        self.web_search_tool = web_search_tool
        
        # Create the tools for CrewAI
        self.tools = [self.vector_db_tool, self.web_search_tool]
        
        # Initialize agents
        self.retriever_agent = self._create_retriever_agent()
        self.writer_agent = self._create_writer_agent()
        self.research_agent = self._create_research_agent()
    
    def _create_retriever_agent(self) -> Agent:
        """Create the Retriever Agent that chooses the appropriate tool."""
        
        return Agent(
            role="Retriever Agent",
            goal="Retrieve the most relevant context for the user query by choosing the best available tool",
            backstory=(
                "You are an expert at information retrieval. Your job is to analyze user queries "
                "and retrieve relevant context using either the vector database or web search. "
                "You choose the most appropriate tool based on the query type."
            ),
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
    
    def _create_writer_agent(self) -> Agent:
        """Create the Writer Agent that generates the final response."""
        
        return Agent(
            role="Writer Agent",
            goal="Generate a clear, accurate, and helpful response based on the retrieved context",
            backstory=(
                "You are an expert writer specializing in technical explanations. "
                "You take the retrieved context and craft a comprehensive, easy-to-understand response. "
                "Always cite your sources when possible."
            ),
            verbose=True,
            allow_delegation=False
        )
    
    def _create_research_agent(self) -> Agent:
        """Create the Research Agent that combines both retrieval methods."""
        
        return Agent(
            role="Research Agent",
            goal="Thoroughly research the user query by retrieving context from both vector database and web search",
            backstory=(
                "You are a research specialist who combines multiple sources of information. "
                "You use both the vector database (for internal knowledge) and web search (for up-to-date information) "
                "to provide comprehensive answers."
            ),
            tools=self.tools,
            verbose=True,
            allow_delegation=False
        )
    
    def run(self, query: str) -> str:
        """Run the complete RAG pipeline."""
        
        # Step 1: Retriever Agent gets context
        retriever_task = Task(
            description=f"Retrieve relevant context for the following query: {query}",
            expected_output="Relevant context from the best available source (vector DB or web search)",
            agent=self.retriever_agent
        )
        
        # Step 2: Writer Agent generates response
        writer_task = Task(
            description="Generate a final response based on the retrieved context",
            expected_output="A clear, accurate, and well-structured response",
            agent=self.writer_agent,
            context=[retriever_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.retriever_agent, self.writer_agent],
            tasks=[retriever_task, writer_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    def research(self, query: str) -> str:
        """Run the Research Agent that uses both retrieval methods."""
        
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
    
    def __init__(self, pipeline: AgenticRAGPipeline):
        self.pipeline = pipeline
    
    def setup(self, device: str = "cuda"):
        """
        Setup method - initializes the RAG pipeline with agents and tools.
        This is where all the CrewAI orchestration is configured.
        """
        # The pipeline is already initialized with agents in __init__
        # This setup method can be used for additional initialization if needed
        print(f"RAG Pipeline initialized on device: {device}")
        print(f"Retriever Agent: {self.pipeline.retriever_agent.role}")
        print(f"Writer Agent: {self.pipeline.writer_agent.role}")
        print(f"Research Agent: {self.pipeline.research_agent.role}")
        print(f"Available tools: {[tool.name for tool in self.pipeline.tools]}")
    
    def decode_request(self, request: Any) -> Dict:
        """Decode the incoming request."""
        return request
    
    def predict(self, inputs: Dict) -> Dict:
        """Run prediction on the inputs."""
        query = inputs.get("query", "")
        mode = inputs.get("mode", "research")  # "research" or "rag"
        
        if mode == "research":
            result = self.pipeline.research(query)
        else:
            result = self.pipeline.run(query)
        
        return {
            "query": query,
            "mode": mode,
            "response": result
        }
    
    def encode_response(self, output: Dict) -> Any:
        """Encode the response."""
        return output


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function to set up and run the LitServe server."""
    
    # Initialize tools
    print("Initializing tools...")
    vector_db_tool = create_sample_vector_db()
    web_search_tool = FirecrawlWebSearchTool()
    
    # Create the RAG pipeline with CrewAI agents
    print("Creating Agentic RAG Pipeline with CrewAI...")
    pipeline = AgenticRAGPipeline(
        vector_db_tool=vector_db_tool,
        web_search_tool=web_search_tool
    )
    
    # Create LitServe API
    api = RAGLitAPI(pipeline)
    
    # Create and start the server
    server = LitServer(api, debug=True)
    print("\n" + "="*50)
    print("Starting Agentic RAG Server...")
    print("="*50)
    print("\nEndpoints:")
    print("  POST /predict - Run RAG pipeline")
    print("\nExample request:")
    print('  {"query": "What is RAG in AI?", "mode": "research"}')
    print("\n" + "="*50 + "\n")
    
    server.run(port=8000)


if __name__ == "__main__":
    main()
