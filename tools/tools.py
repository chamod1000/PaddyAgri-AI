"""
Agent Tools Module
Implements Mandatory Requirement 4a: Tool-Use Agentic Design Pattern
Provides tools for RAG vector search retrieval and fertilizer dosage calculation.
"""

from typing import Dict, List, Any
import streamlit as st
from langchain_core.tools import tool
from core.agent_messages import RAGContextChunk


@st.cache_resource(show_spinner=False)
def get_cached_vector_store():
    """Caches the FAISS vector database to ensure embeddings load instantly across executions."""
    from rag.rag_pipeline import create_or_load_vector_store
    return create_or_load_vector_store(force_rebuild=False)


@tool
def rag_search_tool(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Searches the paddy farming FAISS vector database for relevant domain knowledge chunks.
    Supports both English and Sinhala search queries.

    Args:
        query: Search query text (disease symptoms, fertilizer rules, policy, seed guidelines).
        top_k: Number of relevant chunks to retrieve.

    Returns:
        List of matching document chunks with filename, category, page, and score.
    """
    try:
        vector_store = get_cached_vector_store()
        results = vector_store.similarity_search_with_score(query, k=top_k)

        chunks = []
        for doc, score in results:
            chunks.append({
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "category": doc.metadata.get("category", "General"),
                "page": doc.metadata.get("page", 0) + 1,
                "score": float(score)
            })
        return chunks
    except Exception as e:
        print(f"[TOOL ERROR] RAG search failed: {e}")
        return []


@tool
def fertilizer_calculator_tool(season: str, district: str, field_size_acres: float = 1.0) -> Dict[str, Any]:
    """
    Calculates recommended Sri Lankan Department of Agriculture NPK fertilizer rates (Urea, TSP, MOP) in kg.

    Args:
        season: 'Yala' or 'Maha'
        district: Sri Lankan district name (e.g. 'Polonnaruwa', 'Anuradhapura', 'Kurunegala', 'Ampara', 'Gampaha')
        field_size_acres: Size of paddy field in acres.

    Returns:
        Recommended dosage breakdown in kg and application timetable.
    """
    season_clean = season.strip().capitalize()

    # Department of Agriculture standard base recommendations per acre
    if season_clean == "Yala":
        # Yala (Dry Season) base dosage per acre (kg)
        urea_per_acre = 50.0
        tsp_per_acre = 25.0
        mop_per_acre = 25.0
    else:
        # Maha (Wet Season) base dosage per acre (kg)
        urea_per_acre = 65.0
        tsp_per_acre = 30.0
        mop_per_acre = 30.0

    total_urea = urea_per_acre * field_size_acres
    total_tsp = tsp_per_acre * field_size_acres
    total_mop = mop_per_acre * field_size_acres

    schedule = [
        f"Basal Application (at land preparation / planting): {total_tsp:.1f} kg TSP + {total_urea * 0.2:.1f} kg Urea",
        f"1st Top Dressing (2 weeks after planting): {total_urea * 0.4:.1f} kg Urea",
        f"2nd Top Dressing (4 weeks after planting): {total_mop * 0.5:.1f} kg MOP",
        f"Panicle Initiation (6 weeks after planting): {total_urea * 0.4:.1f} kg Urea + {total_mop * 0.5:.1f} kg MOP"
    ]

    return {
        "season": season_clean,
        "district": district,
        "field_size_acres": field_size_acres,
        "urea_kg": round(total_urea, 2),
        "tsp_kg": round(total_tsp, 2),
        "mop_kg": round(total_mop, 2),
        "schedule": schedule
    }
