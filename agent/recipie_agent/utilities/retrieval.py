import json
from typing import Any, Dict, List, Optional

def retrieve_similar_recipes(
    query_str: str,
    embedder,
    collection_name: str,
    result_limit: int = 1,
    save_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run a vector search in Qdrant and (optionally) save compact results.

    Returns: [{"score": float, "payload": {...}}, ...]
    """
    query_vector = embedder.embed_query(query_str)
    recipe_results = embedder.return_query_results(
        query_vector=query_vector,
        collection_name=collection_name,
        limit_res=result_limit
    )

    if not getattr(recipe_results, "points", None):
        retrieved_data: List[Dict[str, Any]] = []
    else:
        retrieved_data = [
            {"score": round(hit.score, 4), "payload": getattr(hit, "payload", {})}
            for hit in recipe_results.points
        ]

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(retrieved_data, f, indent=2, ensure_ascii=False)
        print(f"📦 Saved {len(retrieved_data)} retrieved recipes to: {save_path}")

    return retrieved_data
