import re

def parse_dishes_block(text: str):
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
    return re.sub(r"[–—−]", ", ", s or "")
