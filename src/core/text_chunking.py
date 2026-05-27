from __future__ import annotations

import re


TOKEN_OR_CJK_RUN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_\-]+")


def tokenize_for_chunking(text: str) -> list[str]:
    """Tokenize English words and CJK characters for stable chunk windows."""

    tokens: list[str] = []
    for match in TOKEN_OR_CJK_RUN_PATTERN.finditer(text):
        token = match.group(0)
        if _is_cjk_run(token):
            tokens.extend(token)
        else:
            tokens.append(token)
    return tokens


def detokenize_chunk(tokens: list[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if not parts:
            parts.append(token)
            continue
        if _is_cjk_token(parts[-1][-1]) and _is_cjk_token(token[0]):
            parts.append(token)
        else:
            parts.append(f" {token}")
    return "".join(parts)


def _is_cjk_token(value: str) -> bool:
    return len(value) == 1 and "\u4e00" <= value <= "\u9fff"


def _is_cjk_run(value: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in value)
