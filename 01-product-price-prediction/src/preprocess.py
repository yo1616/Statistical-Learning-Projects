import re
import unicodedata
from typing import Iterable, List

_unit_alias = {
    "公克": "g", "克": "g", "Ｇ": "g", "ｇ": "g",
    "公斤": "kg", "千克": "kg",
    "公升": "l", "毫升": "ml",
    "包": "pack", "入": "pcs",
    "盎司": "oz", "磅": "lb"
}

def _to_halfwidth(s: str) -> str:
    return unicodedata.normalize('NFKC', s)

def _normalize_units(s: str) -> str:
    for k, v in _unit_alias.items():
        s = s.replace(k, v)
    return s

def _basic_cleanup(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,，]+", ",", s)
    s = re.sub(r"[\-–—]+", "-", s)
    return s

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _to_halfwidth(s)
    s = s.lower()
    s = _normalize_units(s)
    s = _basic_cleanup(s)
    return s

def normalize_texts(xs: Iterable[str]) -> List[str]:
    return [normalize_text(x) for x in xs]
