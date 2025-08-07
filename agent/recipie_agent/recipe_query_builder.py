def build_query_string(expiring, current):
    # Step 1: Collect names of expiring ingredients
    expiring_items = []
    for item in expiring:
        if isinstance(item, dict):
            name = item.get("item") or item.get("name")
            if name:
                expiring_items.append(name)

    # Step 2: Collect names of available inventory items
    inventory_items = []
    for item in current:
        if isinstance(item, dict):
            name = item.get("item") or item.get("name")
            if name:
                inventory_items.append(name)

    # Step 3: Compose full prompt (system + ingredients)
    system_prompt = (
        "You are a professional chef tasked with planning a realistic multi-course meal using the provided ingredients. "
        "Generate 3 distinct dishes using expiring items as a priority and available ingredients to complement them. "
        "Avoid poetic or vague language. Focus on real Mediterranean bistro-style dishes using professional techniques. "
        "Do not list ingredients separately — integrate them into the step-by-step cooking instructions. "
        "Avoid precise measurements — focus on feel, texture, and flavor balance. "
        "Format the result as:\n\n"
        "1. Dish Name\n"
        "   Step-by-step cooking instructions.\n\n"
        "2. Dish Name\n"
        "   Step-by-step cooking instructions.\n\n"
        "3. Dish Name\n"
        "   Step-by-step cooking instructions.\n"
    )

    # Step 4: Attach ingredient info
    ingredient_context = (
        f"\nExpiring ingredients:\n{', '.join(expiring_items)}\n\n"
        f"Available ingredients:\n{', '.join(inventory_items)}\n"
    )

    return f"{system_prompt}\n\n{ingredient_context}"
