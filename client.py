"""
Client code to invoke the Agentic RAG API using requests library.
"""
import requests
import json
from typing import Dict, Optional


# Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/predict"


def query_rag(
    query: str,
    mode: str = "research",
    verbose: bool = True
) -> Dict:
    """
    Send a query to the RAG API.
    
    Args:
        query: The user's question
        mode: Pipeline mode - "research", "rag", or "full"
        verbose: Whether to print the response
    
    Returns:
        Dict containing the API response
    """
    # Prepare request payload
    payload = {
        "query": query,
        "mode": mode
    }
    
    if verbose:
        print(f"\n📤 Sending request...")
        print(f"   Query: {query}")
        print(f"   Mode: {mode}")
        print("-" * 50)
    
    try:
        # Send POST request to the API
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Raise exception for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        if verbose:
            print(f"\n📥 Response received:")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Mode: {result.get('mode', 'N/A')}")
            print(f"   Response length: {result.get('metadata', {}).get('response_length', 0)} chars")
            print("-" * 50)
            print(f"\n💬 Generated Response:")
            print(result.get("response", ""))
        
        return result
        
    except requests.exceptions.ConnectionError:
        error_msg = f"❌ Error: Could not connect to {API_BASE_URL}. Is the server running?"
        if verbose:
            print(error_msg)
        return {"error": error_msg}
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"❌ HTTP Error: {e}"
        if verbose:
            print(error_msg)
        return {"error": error_msg}
        
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Request Error: {e}"
        if verbose:
            print(error_msg)
        return {"error": error_msg}


def query_full_pipeline(query: str, verbose: bool = True) -> Dict:
    """
    Convenience function to use the full pipeline (Research + Writer).
    
    Args:
        query: The user's question
        verbose: Whether to print the response
    
    Returns:
        Dict containing the API response
    """
    return query_rag(query=query, mode="full", verbose=verbose)


def query_research_only(query: str, verbose: bool = True) -> Dict:
    """
    Convenience function to use only the research pipeline.
    
    Args:
        query: The user's question
        verbose: Whether to print the response
    
    Returns:
        Dict containing the API response
    """
    return query_rag(query=query, mode="research", verbose=verbose)


def query_rag_pipeline(query: str, verbose: bool = True) -> Dict:
    """
    Convenience function to use the standard RAG pipeline (Retriever + Writer).
    
    Args:
        query: The user's question
        verbose: Whether to print the response
    
    Returns:
        Dict containing the API response
    """
    return query_rag(query=query, mode="rag", verbose=verbose)


# ============================================================
# Interactive Demo
# ============================================================

def run_demo():
    """Run an interactive demo of the RAG API."""
    
    print("\n" + "=" * 60)
    print("🤖 Agentic RAG Client Demo")
    print("=" * 60)
    print(f"\nAPI Endpoint: {API_ENDPOINT}")
    print("\nAvailable modes:")
    print("  - research: Use Research Agent only")
    print("  - rag: Use Retriever + Writer agents")
    print("  - full: Use Research + Writer agents (complete workflow)")
    print("\n" + "=" * 60)
    
    # Example queries
    example_queries = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "How does CrewAI work?",
        "What are the benefits of using local LLMs?"
    ]
    
    print("\n📝 Example queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"   {i}. {q}")
    
    # Use full pipeline as default
    print("\n" + "-" * 60)
    query = example_queries[0]
    print(f"\n🔍 Running query: {query}")
    print("-" * 60)
    
    # Run the query
    result = query_full_pipeline(query, verbose=True)
    
    return result


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use command line argument as query
        query = " ".join(sys.argv[1:])
        mode = "full"
        
        # Check for mode flag
        if "--mode" in sys.argv:
            mode_idx = sys.argv.index("--mode") + 1
            if mode_idx < len(sys.argv):
                mode = sys.argv[mode_idx]
        
        query_rag(query, mode=mode)
    else:
        # Run demo
        run_demo()
