"""Word-level comparison of two transcripts (sent vs re-transcribed at the ranger).

normalize -> lowercase, strip punctuation, spell out small numbers ("4" == "four"),
then difflib alignment gives equal / replace / delete / insert spans, a word error
rate (WER) and a similarity score, and a plain verdict.
"""
from __future__ import annotations

import difflib
import re

_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
        "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty"}
VERIFIED, PARTIAL, MISMATCH = "verified", "partly matches", "mismatch"


def normalize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))
    out = []
    for w in words:
        w = _NUM.get(w, w)
        out.append(w.replace("'", ""))
    # "northridge" vs "north ridge": compare on joined compounds by splitting nothing; handled by char-level
    # fallback below for near-identical words
    return out


def _close(a: str, b: str) -> bool:
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def compare(sent: str, received: str) -> dict:
    """Align `sent` (reference) against `received` (hypothesis) word by word."""
    a, b = normalize(sent), normalize(received)
    # merge split compounds on either side ("north ridge" ~ "northridge") so they align as one word
    a, b = _merge_compounds(a, b), _merge_compounds(b, a)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops, subs, dels, ins, equal = [], 0, 0, 0, 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        sa, sb = a[i1:i2], b[j1:j2]
        if tag == "replace" and len(sa) == len(sb) and all(_close(x, y) for x, y in zip(sa, sb)):
            tag = "equal"  # spelling-level differences only (e.g. "northridge" vs "north-ridge")
        if tag == "equal":
            equal += len(sa)
        elif tag == "replace":
            subs += min(len(sa), len(sb))
            dels += max(0, len(sa) - len(sb))
            ins += max(0, len(sb) - len(sa))
        elif tag == "delete":
            dels += len(sa)
        elif tag == "insert":
            ins += len(sb)
        ops.append({"op": tag, "sent": " ".join(sa), "received": " ".join(sb)})
    n = max(len(a), 1)
    wer = (subs + dels + ins) / n
    similarity = equal / max(len(a), len(b), 1)
    verdict = VERIFIED if similarity >= 0.85 else PARTIAL if similarity >= 0.6 else MISMATCH
    return {"similarity": round(similarity, 3), "wer": round(wer, 3), "substitutions": subs, "deletions": dels,
            "insertions": ins, "sent_words": len(a), "received_words": len(b), "ops": ops, "verdict": verdict}


def _merge_compounds(words: list[str], other: list[str]) -> list[str]:
    """Join adjacent words that appear as a single word in `other` ("north","ridge" -> "northridge")."""
    targets = set(other)
    out, i = [], 0
    while i < len(words):
        if i + 1 < len(words) and words[i] + words[i + 1] in targets:
            out.append(words[i] + words[i + 1])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return out
