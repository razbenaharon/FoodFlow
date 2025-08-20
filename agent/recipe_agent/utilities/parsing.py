import re


def parse_dishes_block(text: str):
    """
    Parses a block of text containing multiple numbered dishes with titles and reasons.

    Uses regular expressions to extract the title, reason, and body for each dish,
    handling different title formatting (e.g., bold or plain).
    """
    text = text.replace("\r\n", "\n")
    items = []
    pat = re.compile(
        r"""
        ^\s*\d+\.\s+
        (?:\*\*(?P<title_bold>.+?)\*\*|(?P<title_plain>[^\n]+))
        (?P<rest>(?:\n(?!\d+\.\s).*)*)
        """,
        re.MULTILINE | re.VERBOSE,
    )
    reason_pat = re.compile(r"^\s*(?:\*\*\s*)?Reason\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    for m in pat.finditer(text.strip()):
        title = (m.group("title_bold") or m.group("title_plain") or "").strip()
        block = m.group("rest") or ""
        r = reason_pat.search(block)
        reason = r.group(1).strip() if r else ""
        body = reason_pat.sub("", block).strip()
        items.append({"title": title, "body": body, "reason": reason})
    return items


def no_en_dashes(s: str) -> str:
    """Replaces various dash characters with a simple comma and space."""
    return re.sub(r"[–—−]", ", ", s or "")


def extract_json_block(text: str) -> str:
    """
    Extracts the first JSON object string from a given text block.

    Raises a ValueError if no JSON object is found.
    """
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in response.")