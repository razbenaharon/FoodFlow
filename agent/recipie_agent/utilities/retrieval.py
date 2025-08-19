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
    Performs a vector similarity search for recipes in a Qdrant collection.

    It uses an embedder to convert the query string into a vector, then searches
    for the most similar recipes. The results can be optionally saved to a JSON file.

    Args:
        query_str (str): The search query text.
        embedder: An object capable of generating embeddings and performing Qdrant searches.
        collection_name (str): The name of the Qdrant collection to search.
        result_limit (int): The number of top results to retrieve.
        save_path (Optional[str]): The file path to save the retrieved results.

    Returns:
        A list of dictionaries, where each dictionary contains the score and payload
        of a retrieved recipe.
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