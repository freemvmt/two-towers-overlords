"""
Streamlit frontend for two-towers document retrieval system.
"""

import os
import requests
import time
from typing import Dict, Optional, Any

import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


def call_api(endpoint: str, method: str = "GET", json_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make API call with error handling."""
    try:
        url = f"{API_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Is the backend running?")
        return {"error": "Connection failed"}
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return {"error": "Timeout"}
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return {"error": str(e)}


def check_api_health() -> bool:
    """Check if API is healthy and return status."""
    health_data = call_api("/health")

    if "error" in health_data:
        return False

    return health_data.get("status") == "healthy"


def search_documents(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Search for documents using the API."""
    return call_api("/search", method="POST", json_data={"query": query, "top_k": top_k})


def get_index_info() -> Dict[str, Any]:
    """Get index information from the API."""
    return call_api("/index-info")


# Streamlit app configuration
st.set_page_config(
    page_title="Marco Polo",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
.search-result {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background-color: #f9f9f9;
}
.score-badge {
    background-color: #4CAF50;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: bold;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# Main app
st.title("🔍 Two Towers Document Retrieval")
st.markdown("**Semantic search through MS Marco documents using trained embeddings**")

# Sidebar for configuration and status
with st.sidebar:
    st.header("⚙️ Configuration")

    # API health check
    st.subheader("🏥 System Status")
    if st.button("Check API Health", use_container_width=True):
        with st.spinner("Checking API health..."):
            is_healthy = check_api_health()
            if is_healthy:
                st.success("✅ API is healthy")
            else:
                st.error("❌ API is not responding")

    # Index information
    st.subheader("📊 Index Information")
    if st.button("Get Index Info", use_container_width=True):
        with st.spinner("Fetching index information..."):
            index_info = get_index_info()
            if "error" not in index_info:
                st.json(index_info)
            else:
                st.error("Failed to fetch index information")

    # Search parameters
    st.subheader("🎯 Search Parameters")
    top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)

# Main search interface
st.header("🔎 Search Documents")

# Search form
with st.form("search_form"):
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., machine learning algorithms, climate change effects, protein folding...",
        help="Enter natural language queries to find relevant documents",
    )

    submitted = st.form_submit_button("Search", use_container_width=True)

# Handle search
if submitted and query.strip():
    with st.spinner(f"Searching for '{query}'..."):
        start_time = time.time()
        search_results = search_documents(query.strip(), top_k)
        search_time = time.time() - start_time

        if "error" not in search_results:
            # Display search metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Results Found", search_results["total_results"])
            with col2:
                st.metric("⚡ Processing Time", f"{search_results['processing_time_ms']:.1f}ms")
            with col3:
                st.metric("🌐 Total Time", f"{search_time * 1000:.1f}ms")

            # Display results
            st.subheader(f"Results for: '{search_results['query']}'")

            if search_results["results"]:
                for i, result in enumerate(search_results["results"], 1):
                    with st.container():
                        st.markdown(
                            f"""
                        <div class="search-result">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <h4 style="margin: 0;">Result #{i}</h4>
                                <span class="score-badge">Score: {result["score"]:.3f}</span>
                            </div>
                            <p style="margin: 0; line-height: 1.5;">{result["content"]}</p>
                            <small style="color: #666;">ID: {result["id"]}</small>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
            else:
                st.info("🔍 No results found for your query. Try different keywords or phrases.")
        else:
            st.error(f"❌ Search failed: {search_results.get('error', 'Unknown error')}")

elif submitted and not query.strip():
    st.warning("⚠️ Please enter a search query.")

# Help section
with st.expander("ℹ️ How to use this search system"):
    st.markdown("""
    ### About Two-Towers Document Retrieval
    
    This system uses a **two-towers neural architecture** to perform semantic document search:
    
    1. **Query Tower**: Encodes your search query into a vector representation
    2. **Document Tower**: Pre-encoded document vectors stored in Redis
    3. **Similarity Search**: Finds documents with vectors most similar to your query
    """)

# Footer
st.markdown("---")
st.markdown(f"Built with ❤️ using Streamlit, FastAPI, and Redis Vector Search | API Endpoint: `{API_URL}`")
