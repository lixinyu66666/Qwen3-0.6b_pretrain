import html, re

TAG_RE = re.compile(r"<[^>]+>")

def clean_stackoverflow(s):
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_generic(s):
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

