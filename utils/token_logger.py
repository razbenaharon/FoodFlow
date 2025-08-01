import os
import tiktoken

# קביעת נתיב תיקיית הבסיס של הפרויקט
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# נתיבי יומן ספירת טוקנים
LOG_DIR = os.path.join(PROJECT_ROOT, "tokens_count")
CHAT_LOG = os.path.join(LOG_DIR, "chat_tokens.txt")
EMBED_LOG = os.path.join(LOG_DIR, "embedding_tokens.txt")


def log_tokens(n: int, category: str = "chat"):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = CHAT_LOG if category == "chat" else EMBED_LOG
    with open(path, "a") as f:
        f.write(f"{n}\n")


def get_total_tokens_used(category: str = "chat") -> int:
    path = CHAT_LOG if category == "chat" else EMBED_LOG
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        lines = f.readlines()
        return sum(int(line.strip()) for line in lines if line.strip().isdigit())


def estimate_cost(chat_tokens: int, embedding_tokens: int) -> float:
    """
    חישוב עלות כוללת:
    - gpt-4o: $0.005 קלט, $0.015 פלט (בהנחה של 50% קלט/פלט)
    - embedding: $0.02 ל-1,000 טוקנים
    """
    input_tokens = chat_tokens / 2
    output_tokens = chat_tokens / 2
    chat_cost = (input_tokens / 1000) * 0.005 + (output_tokens / 1000) * 0.015
    embedding_cost = (embedding_tokens / 1000) * 0.02
    return round(chat_cost + embedding_cost, 4)


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
