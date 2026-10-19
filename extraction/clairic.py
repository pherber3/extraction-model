"""Cached client for Cloudflare Workers AI `@cf/cloudflare/clairic`."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
N_QUANTILES = 99
_QUANTILE_KEYS = ("quantiles", "percentiles")


class ClairicError(RuntimeError):
    """Non-retryable API failure, exhausted retries, or an unexpected response shape."""


def build_payload(request: Mapping[str, Any], include_quantiles: bool = True) -> dict[str, Any]:
    """Clairic request body from a contract `request` ({state, fields, reference_date?, images?})."""
    payload: dict[str, Any] = {"model": "clairic", "state": request["state"], "fields": request["fields"]}
    if request.get("reference_date") is not None:
        payload["reference_date"] = request["reference_date"]
    if request.get("images"):
        payload["images"] = request["images"]
    payload["options"] = {"include_quantiles": bool(include_quantiles), "return_state_text": False}
    return payload


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def unwrap_envelope(data: Any) -> dict[str, Any]:
    """Return the model response from a Cloudflare envelope ({result, success, errors}) or a bare response."""
    if not isinstance(data, dict):
        raise ClairicError(f"unexpected response type {type(data).__name__}")
    if "result" in data and ("success" in data or "errors" in data):
        if data.get("success") is False:
            raise ClairicError(f"Cloudflare API error: {data.get('errors')}")
        result = data["result"]
        if not isinstance(result, dict):
            raise ClairicError(f"unexpected result type {type(result).__name__}")
        return result
    return data


def _quantile_level(key: Any) -> float:
    s = str(key).strip().lower().lstrip("pq")
    level = float(s)
    return level / 100.0 if level >= 1.0 else level


def normalize_quantiles(q: Any) -> list[float]:
    """Coerce a 99-percentile payload (list, or dict keyed by level) to floats at levels 0.01..0.99."""
    if isinstance(q, Mapping):
        items = sorted((_quantile_level(k), float(v)) for k, v in q.items())
        levels = [lv for lv, _ in items]
        expected = [i / 100 for i in range(1, N_QUANTILES + 1)]
        if len(levels) != N_QUANTILES or any(abs(a - b) > 1e-9 for a, b in zip(levels, expected)):
            raise ClairicError(f"quantile dict does not cover levels 0.01..0.99: {levels[:3]}...")
        return [v for _, v in items]
    if isinstance(q, (list, tuple)):
        if len(q) != N_QUANTILES:
            raise ClairicError(f"expected {N_QUANTILES} quantiles, got {len(q)}")
        return [float(v) for v in q]
    raise ClairicError(f"unexpected quantiles type {type(q).__name__}")


def normalize_answers(answers: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of `answers` with any `quantiles`/`percentiles` field moved to a 99-float `quantiles` list."""
    out: dict[str, Any] = {}
    for name, ans in answers.items():
        ans = dict(ans)
        for key in _QUANTILE_KEYS:
            if ans.get(key) is not None:
                ans["quantiles"] = normalize_quantiles(ans.pop(key))
                break
        ans.pop("percentiles", None)
        out[name] = ans
    return out


class ClairicClient:
    """POSTs extraction requests to Clairic with retry/backoff and a JSONL response cache.

    Cache lines: {key, request, response, fetched_at, latency_s}; key = sha256 of the canonical
    JSON of {"endpoint": model, "body": request body}. `response` is the unwrapped, raw model response.
    """

    def __init__(
        self,
        account_id: str,
        api_token: str,
        cache_path: str | os.PathLike[str],
        model: str = "@cf/cloudflare/clairic",
        timeout: float = 60,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.url = API_BASE.format(account_id=account_id, model=model)
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._cache[entry["key"]] = entry

    @classmethod
    def from_env(cls, cache_path: str | os.PathLike[str], **kwargs: Any) -> "ClairicClient":
        missing = [k for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN") if not os.environ.get(k)]
        if missing:
            raise ClairicError(f"missing environment variables: {', '.join(missing)}")
        return cls(os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"], cache_path, **kwargs)

    def cache_key(self, payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(canonical_json({"endpoint": self.model, "body": payload}).encode("utf-8")).hexdigest()

    def _backoff(self, attempt: int, resp: requests.Response | None) -> float:
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return min(self.backoff_max, float(retry_after))
                except ValueError:
                    pass  # HTTP-date form: fall back to exponential backoff
        return min(self.backoff_max, self.backoff_base * 2.0**attempt)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        last_error = ""
        for attempt in range(self.max_retries + 1):
            resp: requests.Response | None = None
            try:
                resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                elif resp.status_code >= 400:
                    raise ClairicError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
                else:
                    return unwrap_envelope(resp.json())
            if attempt < self.max_retries:
                time.sleep(self._backoff(attempt, resp))
        raise ClairicError(f"giving up after {self.max_retries + 1} attempts; last error: {last_error}")

    def run_raw(self, request: Mapping[str, Any], include_quantiles: bool = True) -> dict[str, Any]:
        """Unwrapped model response ({model, answers, truncated, usage}) as returned by the API or cache."""
        payload = build_payload(request, include_quantiles)
        key = self.cache_key(payload)
        hit = self._cache.get(key)
        if hit is not None:
            return copy.deepcopy(hit["response"])
        t0 = time.perf_counter()
        response = self._post(payload)
        latency = time.perf_counter() - t0
        entry = {
            "key": key,
            "request": payload,
            "response": response,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "latency_s": latency,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(entry) + "\n")
        self._cache[key] = entry
        return copy.deepcopy(response)

    def run(self, request: Mapping[str, Any], include_quantiles: bool = True) -> dict[str, Any]:
        """Model response with answers' quantiles normalized to a 99-float `quantiles` list."""
        response = self.run_raw(request, include_quantiles)
        if "answers" not in response:
            raise ClairicError(f"response has no 'answers': keys={sorted(response)}")
        response["answers"] = normalize_answers(response["answers"])
        return response

    def run_many(
        self, examples: Iterable[Mapping[str, Any]], include_quantiles: bool = True
    ) -> dict[str, dict[str, Any]]:
        """{example id: answers} for contract examples, sequentially (cache makes reruns free)."""
        return {ex["id"]: self.run(ex["request"], include_quantiles)["answers"] for ex in examples}
