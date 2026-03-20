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
from pydantic import BaseModel, Field, validator
from litserve import LitAPI, LitServer


# ============================================================
# Request/Response Models
# ============================================================

class RAGRequest(BaseModel):
    """Request model for the RAG pipeline."""
    query: str = Field(..., description="The user's question or query")
    mode: str = Field(default="research", description="Pipeline mode: 'research', 'rag', or 'full'")
    
    @validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ['research', 'rag', 'full']
        if v not in allowed_modes:
            raise ValueError(f'mode must be one of {allowed_modes}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is Retrieval-Augmented Generation?",
                "mode": "full"
            }
        }


class RAGResponse(BaseModel):
    """Response model for the RAG pipeline."""
    query: str = Field(..., description="The original user query")
    mode: str = Field(..., description="The pipeline mode used")
    response: str = Field(..., description="The generated response")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is RAG?",
                "mode": "full",
                "response": "Retrieval-Augmented Generation (RAG) is..."
            }
        }

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
# Writer Agent and Task Definition
# ============================================================

def create_writer_agent_and_task(llm: LLM) -> tuple[Agent, Task]:
    """
    Create the Writer Agent and Task.
    
    This Agent accepts the insights from the Research Agent and generates
    a polished response based on the retrieved context.
    
    Args:
        llm: The language model to use
    
    Returns:
        Tuple of (Agent, Task)
    """
    # Create the Writer Agent
    writer_agent = Agent(
        role="Writer Agent",
        goal="Generate a clear, accurate, and helpful response based on the insights from the Research Agent",
        backstory=(
            "You are an expert technical writer specializing in crafting clear explanations. "
            "You take the insights and context provided by the Research Agent and transform them "
            "into a polished, well-structured response. Your job is to accept the research findings "
            "and generate a final answer that addresses the user's query comprehensively."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Create the Writer Task
    writer_task = Task(
        description=(
            "Generate a final response based on the insights from the Research Agent:\n\n"
            "1. Review the research findings and context provided.\n"
            "2. Craft a clear, well-structured answer to the user's query.\n"
            "3. Ensure the response is accurate, helpful, and cites sources when possible.\n\n"
            "Input: Research insights and context from the Research Agent\n"
            "Output: Polished final response"
        ),
        expected_output="A clear, accurate, and well-structured response addressing the user's query",
        agent=writer_agent
    )
    
    return writer_agent, writer_task


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
        
        # Create Retriever Agent
        self.retriever_agent = self._create_retriever_agent()
        
        # Create Research Agent and Task explicitly
        self.research_agent, self.research_task = create_research_agent_and_task(
            vector_db_tool=vector_db_tool,
            web_search_tool=web_search_tool,
            llm=self.llm
        )
        
        # Create Writer Agent and Task explicitly
        self.writer_agent, self.writer_task = create_writer_agent_and_task(
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
    
    def run_full_pipeline(self, query: str) -> str:
        """
        Run the full pipeline: Research Agent → Writer Agent.
        
        The Research Agent retrieves context using vector DB and web search.
        The Writer Agent then generates a response based on the research insights.
        """
        # Create research task
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
        
        # Create writer task that uses research context
        writer_task = Task(
            description=(
                f"Based on the research findings below, generate a polished response to the user's query.\n\n"
                f"User Query: {query}\n\n"
                f"Research Insights: Use the context from the Research Agent task above.\n"
                f"Generate a clear, well-structured response that addresses the query."
            ),
            expected_output="A clear, accurate, and well-structured response",
            agent=self.writer_agent,
            context=[research_task]  # Writer receives insights from Research
        )
        
        # Run crew with both agents
        crew = Crew(
            agents=[self.research_agent, self.writer_agent],
            tasks=[research_task, writer_task],
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
        # Crew instances for orchestration
        self.research_crew = None
        self.rag_crew = None
        self.full_research_crew = None
    
    def setup(self, device: str = "cuda"):
        """
        Setup method - initializes the CrewAI orchestration with all agents and tools.
        This is the main entry point for initializing the pipeline.
        
        The setup includes:
        1. Vector Database Tool (FAISS-based retrieval)
        2. Firecrawl Web Search Tool
        3. Retriever Agent - chooses best tool for retrieval
        4. Research Agent - retrieves from both vector DB and web search
        5. Writer Agent - generates final response based on research insights
        
        The Research Agent and Task are explicitly defined to accept user queries
        and retrieve context using the vectorDB tool and Firecrawl web search tool.
        
        The Writer Agent and Task are explicitly defined to accept insights from
        the Research Agent and generate a polished response.
        
        CrewAI Crew Orchestration:
        - Research Crew: Research Agent for context retrieval
        - RAG Crew: Retriever + Writer agents
        - Full Research Crew: Research + Writer agents for complete workflow
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
        
        # Explicitly show Writer Agent and Task info
        print(f"\n📋 Writer Agent Configuration:")
        print(f"   Role: {self.crew.writer_agent.role}")
        print(f"   Goal: {self.crew.writer_agent.goal}")
        print(f"   Task: {self.crew.writer_task.description[:100]}...")
        
        # ============================================================
        # Create CrewAI Crews (Orchestration)
        # ============================================================
        print(f"\n🚀 Setting up CrewAI Crews (Orchestration)...")
        
        # Crew 1: Research Crew (Research Agent only)
        self.research_crew = Crew(
            agents=[self.crew.research_agent],
            tasks=[self.crew.research_task],
            verbose=True
        )
        print("   ✓ Research Crew created (Research Agent)")
        
        # Crew 2: RAG Crew (Retriever + Writer)
        self.rag_crew = Crew(
            agents=[self.crew.retriever_agent, self.crew.writer_agent],
            tasks=[],  # Tasks are created dynamically in run_rag
            verbose=True
        )
        print("   ✓ RAG Crew created (Retriever + Writer Agents)")
        
        # Crew 3: Full Research Crew (Research + Writer)
        self.full_research_crew = Crew(
            agents=[self.crew.research_agent, self.crew.writer_agent],
            tasks=[],  # Tasks are created dynamically in run_full_pipeline
            verbose=True
        )
        print("   ✓ Full Research Crew created (Research + Writer Agents)")
        
        print(f"\n📋 Crew Configuration:")
        print(f"   Research Crew: {len(self.research_crew.agents)} agent(s), {len(self.research_crew.tasks)} task(s)")
        print(f"   RAG Crew: {len(self.rag_crew.agents)} agent(s)")
        print(f"   Full Research Crew: {len(self.full_research_crew.agents)} agent(s)")
        
        print(f"\n{'='*60}")
        print("✅ Setup complete!")
        print(f"   LLM: Ollama ({os.getenv('OLLAMA_MODEL', 'qwen3')})")
        print(f"   Tools: {[tool.name for tool in self.crew.tools]}")
        print(f"   Crews: Research, RAG, Full Research")
        print(f"{'='*60}\n")
    
    def decode_request(self, request: Any) -> RAGRequest:
        """
        Decode the incoming request and extract user query.
        
        Expected request body format:
        {
            "query": "user's question",
            "mode": "research" | "rag" | "full"  (optional, defaults to "research")
        }
        
        Args:
            request: The incoming request (dict or object)
        
        Returns:
            RAGRequest containing extracted query and mode
        """
        # Handle both dict and object requests
        if hasattr(request, 'json'):
            # It's a FastAPI/Starlette request object
            try:
                body = request.json()
            except:
                body = {}
        elif isinstance(request, dict):
            body = request
        else:
            body = {}
        
        # Validate and create RAGRequest model
        try:
            rag_request = RAGRequest(**body)
            print(f"\n📥 Request received:")
            print(f"   Query: {rag_request.query[:50]}..." if len(rag_request.query) > 50 else f"   Query: {rag_request.query}")
            print(f"   Mode: {rag_request.mode}")
            return rag_request
        except Exception as e:
            # Return default request if validation fails
            print(f"\n⚠️ Request validation warning: {e}")
            return RAGRequest(query=body.get("query", ""), mode=body.get("mode", "research"))
    
    def predict(self, inputs: RAGRequest) -> RAGResponse:
        """
        Run prediction on the inputs.
        
        This method is called after decode_request() extracts the query from the request body.
        It passes the decoded user query to the appropriate CrewAI workflow to generate a response.
        
        Args:
            inputs: RAGRequest containing extracted query and mode from decode_request()
        
        Returns:
            RAGResponse containing the generated response from the model
        """
        # Extract the user query from the decoded request
        query = inputs.query
        mode = inputs.mode
        
        if not query:
            return RAGResponse(
                query="",
                mode=mode,
                response=""
            )
        
        try:
            print(f"\n🔄 Running {mode} pipeline with query: {query[:50]}...")
            
            # Pass the user query to the appropriate CrewAI workflow
            # Each crew will use the query to retrieve context and generate response
            if mode == "full":
                # Full pipeline: Pass query to Research Agent → Writer Agent workflow
                result = self.crew.run_full_pipeline(query)
            elif mode == "research":
                # Research pipeline: Pass query to Research Agent
                result = self.crew.run_research(query)
            else:
                # RAG pipeline: Pass query to Retriever Agent → Writer Agent
                result = self.crew.run_rag(query)
            
            print(f"✅ Response generated successfully")
            
            # Return the generated response
            return RAGResponse(
                query=query,
                mode=mode,
                response=str(result)
            )
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return RAGResponse(
                query=query,
                mode=mode,
                response=f"Error: {str(e)}"
            )
    
    def encode_response(self, output: RAGResponse) -> Dict:
        """
        Post-process the response and send it back to the client.
        
        This method is called after predict() generates the response.
        It can be used to:
        - Format the response for the client
        - Add metadata
        - Post-process the generated text
        - Add caching headers
        
        Note: LitServe internally invokes these methods in order:
              decode_request → predict → encode_response
        
        Args:
            output: RAGResponse containing the generated response from CrewAI
        
        Returns:
            Dict representation of the response to send to client
        """
        # Post-process the response
        response_text = output.response
        
        # Example post-processing:
        # - Trim whitespace
        # - Add metadata
        # - Format JSON
        
        # Create the final response dict
        response_dict = {
            "query": output.query,
            "mode": output.mode,
            "response": response_text.strip() if response_text else "",
            "status": "success" if response_text else "empty"
        }
        
        # Add additional metadata
        response_dict["metadata"] = {
            "response_length": len(response_text) if response_text else 0,
            "pipeline_mode": output.mode
        }
        
        print(f"\n📤 Sending response to client:")
        print(f"   Status: {response_dict['status']}")
        print(f"   Response length: {response_dict['metadata']['response_length']} chars")
        
        return response_dict


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
