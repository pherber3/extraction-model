"""Typed value normalizers shared by the synthetic generator and the scorer.

`normalize_value` maps a raw surface string (or an already-typed value) to a
canonical typed value:

- number / duration / estimate / forecast -> float (int when ``decimals == 0``)
- date -> ``"YYYY-MM-DD"``
- time -> ``"HH:MM"`` (``"HH:MM:SS"`` when seconds are non-zero)
- datetime -> ``"YYYY-MM-DDTHH:MM"`` (``":SS"`` appended when non-zero)
- text -> NFKC-normalized, whitespace-collapsed string (compare with `text_equal`)
- list / ``list[<items type>]`` -> list of normalized items

Unparseable input raises ``ValueError``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from typing import Any

NUMERIC_TYPES = frozenset({"number", "duration", "estimate", "forecast"})

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MON = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_NUMERIC_DATE = re.compile(r"(?<![\d.,])(\d{1,2})([/.\-])(\d{1,2})\2(\d{4}|\d{2})(?![\d,]|\.\d)")
_MONTH_FIRST = re.compile(
    rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})(?!\d))?", re.IGNORECASE
)
_DAY_FIRST = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\b\.?(?:,?\s+(\d{{4}})(?!\d))?",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"(?<![\d:])(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?|(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])",
    re.IGNORECASE,
)
# Digits with optional thousands/decimal separators; a space only counts as a
# thousands separator when followed by exactly three digits.
_NUMBER = re.compile(r"\d+(?:(?:[.,']|[ \u00a0\u202f](?=\d{3}(?!\d)))\d+)*")
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_TEXT_EDGE = " \t\n.,;:!?\"'`()[]{}"


def normalize_value(
    type_: str, value: Any, decimals: int | None = None, reference_date: str | None = None
) -> Any:
    """Normalize ``value`` (surface string or typed value) to the canonical value for ``type_``."""
    if value is None:
        raise ValueError("cannot normalize None")
    t = type_.strip().lower()
    if t == "text":
        return normalize_text(value)
    if t in NUMERIC_TYPES:
        return _round(parse_number(value), decimals)
    if t == "date":
        return parse_date(value, reference_date).isoformat()
    if t == "time":
        return _fmt_time(parse_time(value))
    if t == "datetime":
        return _fmt_datetime(parse_datetime(value, reference_date))
    if t == "list" or (t.startswith("list[") and t.endswith("]")):
        item_type = t[5:-1].strip() if t != "list" else "text"
        items = value if isinstance(value, (list, tuple)) else _split_list(str(value))
        return [normalize_value(item_type, x, decimals, reference_date) for x in items]
    raise ValueError(f"unknown field type {type_!r}")


def text_equal(a: str, b: str) -> bool:
    """Case-, whitespace-, and edge-punctuation-insensitive text comparison."""
    return _text_key(a) == _text_key(b)


def normalize_text(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean is not text")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = unicodedata.normalize("NFKC", str(value))
    return " ".join(s.split())


def _text_key(value: Any) -> str:
    return normalize_text(value).casefold().strip(_TEXT_EDGE)


def parse_number(value: Any) -> float:
    """Parse a money/number string ("$1,080.00", "USD 1080", "1.080,00 EUR", "2.4 kg")."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    if isinstance(value, (int, float)):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"non-finite number {value!r}")
        return x
    s = unicodedata.normalize("NFKC", str(value)).strip()
    m = _NUMBER.search(s)
    if m is None:
        word = s.casefold().strip(_TEXT_EDGE)
        if word in _NUMBER_WORDS:
            return float(_NUMBER_WORDS[word])
        raise ValueError(f"no number in {value!r}")
    prefix = re.sub(r"[^\-\u2212(]", "", s[: m.start()])
    negative = prefix.endswith(("-", "\u2212", "("))
    x = _parse_number_token(m.group(0))
    return -x if negative else x


def _parse_number_token(token: str) -> float:
    tok = re.sub(r"[ \u00a0\u202f']", "", token)
    has_dot, has_comma = "." in tok, "," in tok
    if has_dot and has_comma:
        dec = "." if tok.rfind(".") > tok.rfind(",") else ","
    elif has_comma:
        parts = tok.split(",")
        dec = "," if len(parts) == 2 and len(parts[1]) != 3 else None
    elif has_dot:
        dec = "." if tok.count(".") == 1 else None
    else:
        dec = None
    if dec is None:
        return float(re.sub(r"[.,]", "", tok))
    i = tok.rfind(dec)
    whole = re.sub(r"[.,]", "", tok[:i]) or "0"
    return float(f"{whole}.{tok[i + 1:]}")


def _round(x: float, decimals: int | None) -> float | int:
    if decimals is None:
        return x
    if decimals == 0:
        return int(round(x))
    return round(x, decimals)


def find_dates(text: str, reference_date: str | None = None) -> list[tuple[int, int, str]]:
    """All non-overlapping parseable dates in ``text`` as ``(start, end, iso)``, leftmost-longest first."""
    ref_year = dt.date.fromisoformat(reference_date).year if reference_date else None
    found: list[tuple[int, int, str]] = []
    for pattern, kind in (
        (_ISO_DATE, "iso"), (_NUMERIC_DATE, "numeric"), (_MONTH_FIRST, "mf"), (_DAY_FIRST, "df")
    ):
        for m in pattern.finditer(text):
            d = _match_to_date(m, kind, ref_year)
            if d is not None:
                found.append((m.start(), m.end(), d.isoformat()))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[tuple[int, int, str]] = []
    for item in found:
        if not out or item[0] >= out[-1][1]:
            out.append(item)
    return out


def _match_to_date(m: re.Match[str], kind: str, ref_year: int | None) -> dt.date | None:
    try:
        if kind == "iso":
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if kind == "numeric":
            a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
            if len(m.group(4)) == 2:
                y += 2000
            # Day-first unless the first component cannot be a day-of-month month pair.
            day, month = (b, a) if b > 12 >= a else (a, b)
            return dt.date(y, month, day)
        if kind == "mf":
            month, day, year = _MONTHS[m.group(1).lower()], int(m.group(2)), m.group(3)
        else:
            day, month, year = int(m.group(1)), _MONTHS[m.group(2).lower()], m.group(3)
        if year is None:
            if ref_year is None:
                return None
            return dt.date(ref_year, month, day)
        return dt.date(int(year), month, day)
    except ValueError:
        return None


def parse_date(value: Any, reference_date: str | None = None) -> dt.date:
    """Parse ISO, d/m/y, m/d/y (when day > 12), and month-name dates; missing year -> reference year."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"not a date: {value!r}")
    s = unicodedata.normalize("NFKC", value)
    found = find_dates(s, reference_date)
    if not found:
        raise ValueError(f"no parseable date in {value!r}")
    return dt.date.fromisoformat(found[0][2])


def parse_time(value: Any) -> dt.time:
    """Parse "14:32", "2:32 PM", "2pm", "14:32:05"."""
    if isinstance(value, dt.datetime):
        return value.time().replace(microsecond=0, tzinfo=None)
    if isinstance(value, dt.time):
        return value.replace(microsecond=0, tzinfo=None)
    if not isinstance(value, str):
        raise ValueError(f"not a time: {value!r}")
    t = _find_time(unicodedata.normalize("NFKC", value))
    if t is None:
        raise ValueError(f"no parseable time in {value!r}")
    return t[2]


def _find_time(s: str) -> tuple[int, int, dt.time] | None:
    for m in _TIME.finditer(s):
        try:
            if m.group(4) is not None:
                hour, minute, sec = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)
                if not 1 <= hour <= 12:
                    continue
                hour = hour % 12 + (12 if m.group(4).lower() == "p" else 0)
            else:
                hour, minute, sec = int(m.group(5)), int(m.group(6)), int(m.group(7) or 0)
            return m.start(), m.end(), dt.time(hour, minute, sec)
        except ValueError:
            continue
    return None


def parse_datetime(value: Any, reference_date: str | None = None) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value.replace(microsecond=0, tzinfo=None)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        raise ValueError(f"not a datetime: {value!r}")
    s = unicodedata.normalize("NFKC", value).strip()
    try:
        return dt.datetime.fromisoformat(s).replace(microsecond=0, tzinfo=None)
    except ValueError:
        pass
    t = _find_time(s)
    if t is None:
        return dt.datetime.combine(parse_date(s, reference_date), dt.time())
    date = parse_date(s[: t[0]] + " " + s[t[1]:], reference_date)
    return dt.datetime.combine(date, t[2])


def _fmt_time(t: dt.time) -> str:
    return t.strftime("%H:%M:%S" if t.second else "%H:%M")


def _fmt_datetime(x: dt.datetime) -> str:
    return x.strftime("%Y-%m-%dT%H:%M:%S" if x.second else "%Y-%m-%dT%H:%M")


def _split_list(s: str) -> list[str]:
    parts = re.split(r"\s*(?:[,;\n]|\band\b)\s*", s)
    return [p for p in (x.strip() for x in parts) if p]
