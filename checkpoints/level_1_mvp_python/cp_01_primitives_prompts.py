"""Checkpoint 01: Primitives, String Manipulation & Prompt Formatting

Repository: github.com/maddox-bayn/aicheck0
Contains 10 production-grade exercises for Checkpoint 01.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, AsyncIterator, Callable, Iterable, Iterator
import ast
import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import time
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# Exercise 01.01: format_system_prompt
def format_system_prompt(role: str, context: str, query: str) -> str:
    """Format prompt string stripping whitespace from all inputs: 'Role: {role}\nContext: {context}\nQuery: {query}'."""
    return f"Role: {role.strip()}\nContext: {context.strip()}\nQuery: {query.strip()}"


# Exercise 01.02: estimate_tokens
def estimate_tokens(text: str) -> int:
    """Estimate token count using 1 token ~= 4 chars, rounded up. Return 0 for empty string."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


# Exercise 01.03: clean_generation
def clean_generation(raw_output: str, stop_token: str = '<|endoftext|>') -> str:
    """Remove all occurrences of stop_token and truncate everything at/after '### END', then strip."""
    if "### END" in raw_output:
        raw_output = raw_output.split("### END", 1)[0]
    raw_output = raw_output.replace(stop_token, "")
    return raw_output.strip()


# Exercise 01.04: calculate_call_cost
def calculate_call_cost(prompt_tokens: int, completion_tokens: int, input_price_per_m: float, output_price_per_m: float) -> float:
    """Compute total USD cost given per-1M-token rates, rounded to 6 decimal places."""
    cost = (prompt_tokens * input_price_per_m + completion_tokens * output_price_per_m) / 1_000_000.0
    return round(cost, 6)


# Exercise 01.05: extract_code_block
def extract_code_block(llm_response: str, language: str = 'python') -> str:
    """Extract raw code inside ```{language} ... ``` or generic ``` ... ``` fences, or stripped string."""
    match = re.search(rf"```{re.escape(language)}\s*\n?(.*?)\n?```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```[^\n]*\n?(.*?)\n?```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_response.strip()


# Exercise 01.06: clamp_temperature
def clamp_temperature(temp: float | int) -> float:
    """Validate numeric type (reject bool/str with TypeError) and clamp to [0.0, 2.0]."""
    if isinstance(temp, bool) or not isinstance(temp, (int, float)):
        raise TypeError("Temperature must be a float or int.")
    return float(max(0.0, min(2.0, float(temp))))


# Exercise 01.07: wrap_untrusted_input
def wrap_untrusted_input(user_text: str, tag: str = 'user_input') -> str:
    """Escape existing <tag> or </tag> as &lt;tag&gt; and &lt;/tag&gt; and wrap in <tag>...</tag>."""
    escaped = user_text.replace(f"<{tag}>", f"&lt;{tag}&gt;").replace(f"</{tag}>", f"&lt;/{tag}&gt;")
    return f"<{tag}>{escaped}</{tag}>"


# Exercise 01.08: contains_flagged_keyword
def contains_flagged_keyword(prompt: str, banned_csv: str) -> bool:
    """Check if prompt contains any whole word from comma-separated banned_csv (case-insensitive)."""
    words = [w.strip() for w in banned_csv.split(",") if w.strip()]
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", prompt, re.IGNORECASE):
            return True
    return False


# Exercise 01.09: middle_truncate
def middle_truncate(text: str, max_chars: int) -> str:
    """If len(text) > max_chars, keep first max_chars//2 and last max_chars//2 joined by '...[TRUNCATED]...'."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}...[TRUNCATED]...{text[len(text)-half:]}"


# Exercise 01.10: parse_model_id
def parse_model_id(model_string: str) -> tuple[str, str, str]:
    """Parse 'provider/model-YYYY-MM-DD' into (provider, model_family, version_date). Default version_date='latest'. Raise ValueError if '/' missing."""
    if "/" not in model_string:
        raise ValueError("Invalid model string: missing '/'")
    provider, rest = model_string.split("/", 1)
    match = re.search(r"-(\d{4}-\d{2}-\d{2})$", rest)
    if match:
        version_date = match.group(1)
        model_family = rest[:match.start()]
    else:
        version_date = "latest"
        model_family = rest
    return provider, model_family, version_date

