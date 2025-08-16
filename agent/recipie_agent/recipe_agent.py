import os
import re
import json
import time
import unicodedata
from typing import Any, Dict, List, Optional

from utils.chat_and_embedding import LLMChat, LLMEmbed
from agent.recipie_agent.utilities.console import LiveBar, Progress, suppress_stdout_stderr, run_in_thread
from agent.recipie_agent.utilities.parsing import parse_dishes_block, no_en_dashes
from agent.recipie_agent.utilities.classify import extract_matched_sets, safe_join, extract_json_block
from agent.recipie_agent.utilities.restaurant import (
    load_top_restaurant, restaurant_capsule, tailor_reason_for_restaurant
)
from agent.recipie_agent.utilities.io_utils import load_json
from agent.recipie_agent.utilities.retrieval import retrieve_similar_recipes
from agent.recipie_agent.utilities.recipe_query_builder import build_query_string

# ======================
# Config
# ======================
TAILOR_REASONS: bool = True
FORCE_RESTAURANT_NAME: str = "HaSalon"
FORCE_RESTAURANT_CONTEXT: str = "Israeli Mediterranean; Tel Aviv; produce driven; live fire; chef Eyal Shani"

OUTPUT_FILENAME = "best_matching_recipes.json"   # only this final file is saved
COLLECTION_NAME = "food_recipes"                 # Qdrant collection
TOP_K = 1                                        # retrieved per dish


# ----------------------
# Normalization & fallback matching (avoid 'none' + de-dupe across buckets)
# ----------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', s.lower()).strip()

def _variants(name: str) -> List[str]:
    n = _norm(name or "")
    out = {n}
    if n.endswith('es'):
        out.add(n[:-2])
    if n.endswith('s'):
        out.add(n[:-1])
    parts = n.split()
    if len(parts) > 1:
        out.update(parts)
    return [v for v in out if v]

def _fallback_match(names: List[str], text: str) -> List[str]:
    t = f" {_norm(text)} "
    hits: List[str] = []
    seen = set()
    for name in names:
        if not name:
            continue
        for v in _variants(name):
            if re.search(rf'(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])', t) or f" {v} " in t:
                key = _norm(name)
                if key not in seen:
                    seen.add(key)
                    hits.append(name)  # keep original form
                break
    return hits

def _subtract_preserve_order(primary: List[str], to_remove: List[str]) -> List[str]:
    """Remove any items from primary whose normalized form appears in to_remove; keep order."""
    remove_keys = {_norm(x) for x in to_remove if isinstance(x, str)}
    out, seen = [], set()
    for x in primary:
        if not isinstance(x, str):
            continue
        k = _norm(x)
        if k in remove_keys:
            continue
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def run_find_recipes() -> None:
    # --- paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
    RECIPES_DIR  = os.path.join(RESULTS_DIR, "recipes")
    RETRIEVED_DIR = os.path.join(RECIPES_DIR, "retrieved")
    CLASS_RAW_DIR = os.path.join(RECIPES_DIR, "classification_raw")
    os.makedirs(RETRIEVED_DIR, exist_ok=True)
    os.makedirs(CLASS_RAW_DIR, exist_ok=True)

    EXPIRING_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")
    INVENTORY_FILE = os.path.join(DATA_DIR, "full_inventory.json")
    CLASS_PROMPT_FILE = os.path.join(PROJECT_ROOT, "agent", "recipie_agent", "prompts", "ingredient_classification_prompt.txt")

    # --- load data
    with suppress_stdout_stderr():
        expiring, _ = load_json(EXPIRING_FILE, "expiring ingredients")
        current,  _ = load_json(INVENTORY_FILE, "current inventory")

    # ✅ lists for fallback matching
    exp_names  = [d.get("item") or d.get("name") for d in expiring if isinstance(d, dict)]
    inv_names  = [d.get("item") or d.get("name") for d in current  if isinstance(d, dict)]

    # --- build query for dish generation
    full_prompt: str = build_query_string(expiring, current)

    # --- progress ui
    TOTAL_UNITS = 60
    progress = Progress(TOTAL_UNITS)

    def _pipeline(advance):
        # Stage A
        advance(2)

        # Stage B: get 3 dishes from GPT
        def _get_three_dishes() -> List[Dict[str, str]]:
            chat_model = LLMChat(temp=0.7)
            with suppress_stdout_stderr():
                resp = chat_model.send_prompt(full_prompt)

            dishes_local = parse_dishes_block(resp)
            if len(dishes_local) != 3:
                for _ in range(2):
                    with suppress_stdout_stderr():
                        resp = chat_model.send_prompt(full_prompt)
                    dishes_local = parse_dishes_block(resp)
                    if len(dishes_local) == 3:
                        break
            if len(dishes_local) != 3:
                raise ValueError("Failed to get 3 dishes from GPT.")
            return dishes_local

        dishes = _get_three_dishes()
        advance(20)

        # Stage C: RAG — retrieve top recipe titles for each dish AND save per-dish JSON
        embed_model = LLMEmbed()

        def _extract_titles_from_obj(objs: Any) -> List[str]:
            out: List[str] = []
            if isinstance(objs, list):
                for r in objs:
                    if isinstance(r, dict):
                        t = (
                            r.get("title") or r.get("name") or
                            (r.get("payload", {}).get("title") if isinstance(r.get("payload"), dict) else None)
                        )
                        if isinstance(t, str):
                            out.append(t.strip())
            elif isinstance(objs, dict):
                if isinstance(objs.get("payload"), dict):
                    t = objs["payload"].get("title")
                    if isinstance(t, str):
                        out.append(t.strip())
                seqs = objs.get("results") or objs.get("hits") or objs.get("matches")
                if isinstance(seqs, list):
                    for r in seqs:
                        if isinstance(r, dict):
                            t = (
                                r.get("title") or r.get("name") or
                                (r.get("payload", {}).get("title") if isinstance(r.get("payload"), dict) else None)
                            )
                            if isinstance(t, str):
                                out.append(t.strip())
            return out

        def _read_top_title_from_disk(path: str) -> Optional[str]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                titles = _extract_titles_from_obj(data)
                return titles[0] if titles else None
            except Exception:
                return None

        def _retrieve_all_top_titles_from_disk() -> List[Optional[str]]:
            """Always read back the saved retrieval files to get the REAL retrieved titles."""
            titles: List[Optional[str]] = []
            for i, d in enumerate(dishes, 1):
                save_path = os.path.join(RETRIEVED_DIR, f"retrieved_dish{i}.json")
                query_text = f"{d['title']}\n{d['body']}"
                with suppress_stdout_stderr():
                    # This saves to disk; we don't rely on the return value.
                    retrieve_similar_recipes(
                        query_str=query_text,
                        embedder=embed_model,
                        collection_name=COLLECTION_NAME,
                        result_limit=TOP_K,
                        save_path=save_path
                    )
                top_title = _read_top_title_from_disk(save_path)
                titles.append(top_title)
                progress.advance(5)
            return titles

        retrieved_titles = _retrieve_all_top_titles_from_disk()

        # Stage D: classify ingredients (in-memory only; no per-dish analysis files)
        with open(CLASS_PROMPT_FILE, "r", encoding="utf-8") as f:
            classification_prompt_template = f.read()

        def _classify_all() -> List[Optional[Dict[str, Any]]]:
            analyses: List[Optional[Dict[str, Any]]] = [None, None, None]
            chat_model_struct = LLMChat(temp=0.2)
            for idx, d in enumerate(dishes, 1):
                dish_text = f"{d['title']}\n{d['body']}"
                filled = (
                    classification_prompt_template
                    .replace("<DISH_TEXT>", dish_text)
                    .replace("<EXPIRING_LIST>",  ", ".join([n for n in exp_names if n]))
                    .replace("<INVENTORY_LIST>", ", ".join([n for n in inv_names if n]))
                )
                with suppress_stdout_stderr():
                    response = chat_model_struct.send_prompt(filled)

                # 💾 save raw for debugging (one file per dish)
                raw_path = os.path.join(CLASS_RAW_DIR, f"dish{idx}_raw.txt")
                try:
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(response)
                except Exception:
                    pass

                try:
                    parsed = json.loads(extract_json_block(response))
                except Exception:
                    parsed = None
                analyses[idx - 1] = parsed
                progress.advance(5)
            return analyses

        analyses_by_dish = _classify_all()

        # Titles: ALWAYS prefer the title read from the saved retrieved files.
        final_titles: List[str] = []
        for i, d in enumerate(dishes):
            rt = retrieved_titles[i]
            final_titles.append(
                rt.strip() if isinstance(rt, str) and rt.strip()
                else (d["title"].strip() or "Untitled")
            )
        reasons = [d.get("reason", "").strip() for d in dishes]

        # finish
        progress.advance(TOTAL_UNITS)
        return final_titles, reasons, analyses_by_dish, dishes

    # drive thread + bar
    bar = LiveBar(prefix="Recipe agent      ", width=30)
    th, res, err = run_in_thread(_pipeline, progress.advance)

    while th.is_alive():
        target = 100.0 * progress.fraction()
        bar.tick_towards(target, step=0.8)
        time.sleep(0.05)

    for _ in range(60):
        bar.tick_towards(100.0, step=2.5)
        if bar.pct >= 100.0:
            break
        time.sleep(0.01)
    bar.finish()

    if err:
        raise err[0]
    final_titles, reasons, analyses_by_dish, dishes = res[0]

    # --- restaurant context (forced to HaSalon unless FORCE_* empty)
    top_rest = load_top_restaurant(RESULTS_DIR)
    rest_name = (FORCE_RESTAURANT_NAME or "").strip()
    rest_capsule = no_en_dashes((FORCE_RESTAURANT_CONTEXT or "").strip())
    if not rest_name and top_rest:
        rest_name = top_rest.get("name") or top_rest.get("title") or "the chosen restaurant"
        rest_capsule = restaurant_capsule(top_rest)

    # --- compose final summary (print for console; save ONLY JSON)
    print("\nBest matching recipes:")

    summary: List[Dict[str, Any]] = []
    for i, title in enumerate(final_titles[:3], 1):
        analysis = analyses_by_dish[i - 1]
        exp_match, inv_match, miss = extract_matched_sets(analysis)

        # 🔁 Fallback if classifier returned nothing
        if not exp_match and not inv_match and not miss:
            dish_text = f"{dishes[i - 1]['title']}\n{dishes[i - 1]['body']}"
            exp_match = _fallback_match(exp_names, dish_text)
            inv_match = _fallback_match(inv_names, dish_text)
            miss = []  # we can't infer missing from text alone

        # ✅ Inventory must exclude expiring (no overlap), and Missing excludes both
        inv_match = _subtract_preserve_order(inv_match, exp_match)
        miss = _subtract_preserve_order(miss, exp_match + inv_match)

        reason = (reasons[i - 1] or "").strip()
        if TAILOR_REASONS and rest_name:
            reason = tailor_reason_for_restaurant(
                dish_title=title,
                expiring_list=exp_match,
                inventory_list=inv_match,
                missing_list=miss,
                restaurant_name=rest_name,
                restaurant_capsule_text=rest_capsule
            )

        print(f"{i}. {title}")
        print(f"   Matched expiring ingredients: {safe_join(exp_match)}")
        print(f"   Inventory ingredients: {safe_join(inv_match)}")
        print(f"   Missing ingredients: {safe_join(miss)}")
        print(f"   Reason: {reason}")

        summary.append({
            "rank": i,
            "title": title,  # ← this is now the REAL retrieved recipe title
            "matched_expiring": exp_match,
            "inventory": inv_match, #TODO: add explanation about none
            "missing": miss,
            "reason": reason,
            "restaurant": rest_name,
            "restaurant_context": rest_capsule,
        })

    # --- save only the final JSON
    json_path = os.path.join(RECIPES_DIR, OUTPUT_FILENAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_find_recipes()
