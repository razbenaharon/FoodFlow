"""
Token accounting utilities and approximate cost estimation.

- `log_tokens`: append token counts to per-category logs
- `get_total_tokens_used`: sum tokens from logs
- `estimate_cost`: rough $ estimation for chat + embeddings
- `count_tokens`: count tokens for a given model (tiktoken)
- `print_estimated_cost_header`: pretty banner printed by main.py
"""
import os
import tiktoken

# Project root (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Token logs live here
LOG_DIR = os.path.join(PROJECT_ROOT, "tokens_count")
CHAT_LOG = os.path.join(LOG_DIR, "chat_tokens.txt")
EMBED_LOG = os.path.join(LOG_DIR, "embedding_tokens.txt")


def log_tokens(n: int, category: str = "chat") -> None:
    """Append `n` tokens to the appropriate log file (`chat` or `embedding`)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = CHAT_LOG if category == "chat" else EMBED_LOG
    with open(path, "a") as f:
        f.write(f"{n}\n")


def get_total_tokens_used(category: str = "chat") -> int:
    """Return the sum of tokens recorded for the given category."""
    path = CHAT_LOG if category == "chat" else EMBED_LOG
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        lines = f.readlines()
        return sum(int(line.strip()) for line in lines if line.strip().isdigit())


def estimate_cost(chat_tokens: int, embedding_tokens: int) -> float:
    """
    Estimate total USD cost using simple assumptions:
    - gpt-4o: $0.0025 per 1K input tokens, $0.01 per 1K output tokens (assumes 50/50 split)
    - embedding-3-small: $0.02 per 1,000,000 tokens
    """
    input_tokens = chat_tokens / 2
    output_tokens = chat_tokens / 2
    chat_cost = (input_tokens / 1000) * 0.0025 + (output_tokens / 1000) * 0.01
    embedding_cost = (embedding_tokens / 1_000_000) * 0.02
    return round(chat_cost + embedding_cost, 5)


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Return the number of tokens in `text` for the specified `model`."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def report_cost() -> None:
    """Print the total estimated cost so far, based on accumulated logs."""
    chat_tokens_used = get_total_tokens_used("chat")
    embedding_tokens_used = get_total_tokens_used("embedding")
    total_cost = estimate_cost(chat_tokens_used, embedding_tokens_used)
    print(f"Total estimated cost: ${total_cost}")


def print_estimated_cost_header() -> None:
    """
    Print a concise, human-readable cost summary for the project based on
    accumulated token usage. Intended to be called from main.py (not on import).
    """
    try:
        chat_tokens = get_total_tokens_used("chat") or 0
        embedding_tokens = get_total_tokens_used("embedding") or 0
        total_cost = estimate_cost(chat_tokens, embedding_tokens)

        print("\n==================== Project Estimated Cost ====================")
        print(f"💬 Chat tokens: {chat_tokens:,}")
        print(f"🔎 Embedding tokens: {embedding_tokens:,}")
        print(f"💲 Total estimated cost: ${total_cost:.5f}")
        print("===============================================================\n")
    except Exception as e:
        print("\n[Cost] Failed to compute estimated cost:", e, "\n")
