"""Text cleaning utilities shared by all three models.

The core `clean_text` mirrors the notebook logic: HTML stripping,
contraction expansion, lower-casing, punctuation/digit removal and
whitespace normalisation. Heavy dependencies (bs4, contractions) are
optional; a regex fallback keeps the package importable without the
`ingest` extra.
"""

from __future__ import annotations

import re
import string

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _strip_html(text: str) -> str:
    try:
        import warnings

        from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

        # A review consisting of a bare URL makes bs4 warn that the input "looks
        # more like a URL than HTML". That is exactly what we are feeding it, on
        # purpose, a million times over — the warning is noise here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
            return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except ImportError:
        return _HTML_TAG_RE.sub(" ", text)


def _expand_contractions(text: str) -> str:
    try:
        import contractions

        return contractions.fix(text)
    except ImportError:
        return text


def clean_text(text: str | None) -> str:
    """Normalise a raw review into model-ready text."""
    if text is None or not isinstance(text, str):
        return ""
    text = _strip_html(text)
    text = _URL_RE.sub(" ", text)
    text = _expand_contractions(text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def basic_text_stats(text: str) -> dict[str, float]:
    """Word count, average word length and lexical diversity for one review."""
    words = text.split()
    if not words:
        return {"word_count": 0, "avg_word_length": 0.0, "lexical_diversity": 0.0}
    return {
        "word_count": len(words),
        "avg_word_length": sum(len(w) for w in words) / len(words),
        "lexical_diversity": len(set(words)) / len(words),
    }
