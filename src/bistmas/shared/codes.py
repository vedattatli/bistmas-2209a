"""Equity code conventions shared by the data layer.

Borsa İstanbul equity codes are 4–5 characters, start with a letter and may contain digits
(A1CAP, A1YEN). Three-letter codes are bank instruments, not equities.
"""

from __future__ import annotations

import re

EQUITY_CODE_PATTERN = r"[A-Z][A-Z0-9]{3,4}"
EQUITY_CODE_RE = re.compile(rf"^{EQUITY_CODE_PATTERN}$")
CORE_INDICES: tuple[str, ...] = ("XU100", "XU050", "XU030")


def is_equity_code(value: str | None) -> bool:
    return bool(value) and EQUITY_CODE_RE.fullmatch(str(value).strip().upper()) is not None
