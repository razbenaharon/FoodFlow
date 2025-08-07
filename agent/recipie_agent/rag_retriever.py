import json

def retrieve_similar_recipes(query_str, embedder, collection_name, result_limit, save_path):
    # Embed query
    query_vector = embedder.embed_query(query_str)

    # Search in Qdrant
    recipe_results = embedder.return_query_results(
        query_vector=query_vector,
        collection_name=collection_name,
        limit_res=result_limit
    )

    if not recipe_results.points:
        print("⚠️ No recipes retrieved from Qdrant.")
        return []

    # Build result list
    retrieved_data = []
    for hit in recipe_results.points:
        retrieved_data.append({
            "score": round(hit.score, 4),
            "payload": hit.payload
        })

    # Save to file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(retrieved_data, f, indent=2, ensure_ascii=False)
    print(f"📦 Saved {len(retrieved_data)} retrieved recipes to: {save_path}")

    return retrieved_data
