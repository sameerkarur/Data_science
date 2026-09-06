# Regular Expressions, JSON Streaming & Web REST APIs
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Why Text Extraction & API Parsing Matter in Data Science](#1-why-text-extraction--api-parsing-matter)
2. [Regular Expressions (Regex) Meta-Characters & Cheat Sheet](#2-regular-expressions-regex-meta-characters)
3. [Python `re` Module: `search()`, `match()`, `findall()`, `sub()`](#3-python-re-module)
4. [Named Capture Groups & Lookarounds (Lookahead & Lookbehind)](#4-named-capture-groups--lookarounds)
5. [JSON Parsing, Nested Extraction & Streaming](#5-json-parsing-nested-extraction--streaming)
6. [RESTful Web APIs & the `requests` Library](#6-restful-web-apis--the-requests-library)
7. [Error Handling, Jittered Exponential Backoff & Rate Limits](#7-error-handling--rate-limits)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Why Text Extraction & API Parsing Matter

In modern machine learning pipelines, over 80% of enterprise information originates from unstructured text, web scrapers, and external third-party HTTP endpoints. Mastering regular expressions and resilient REST API consumption is an indispensable prerequisite for feature extraction.

---

## 2. Regular Expressions (Regex) Meta-Characters

```
  ┌──────────┬─────────────────────────────────────┬────────────────────────┐
  │ Pattern  │ Description                         │ Example Match          │
  ├──────────┼─────────────────────────────────────┼────────────────────────┤
  │ ^        │ Start of string / line              │ ^https                 │
  │ $        │ End of string / line                │ \.csv$                 │
  │ \d       │ Any digit [0-9]                     │ \d{4} (4-digit year)   │
  │ \w       │ Any alphanumeric character [a-zA-Z0-9_]│ \w+                 │
  │ \s       │ Any whitespace (space, tab, newline)│ \s+                    │
  │ +        │ 1 or more repetitions               │ a+                     │
  │ *        │ 0 or more repetitions               │ a*                     │
  │ ?        │ 0 or 1 repetition (or non-greedy)   │ https?                 │
  │ [a-z]    │ Character set                       │ [A-Z0-9]               │
  │ (?P<name>)│ Named capture group                │ (?P<id>\d+)            │
  └──────────┴─────────────────────────────────────┴────────────────────────┘
```

---

## 3. Python `re` Module: Core Functions

```python
import re

log_line = "2026-09-06 10:14:22 [ERROR] UserID: 8492 failed login from IP: 192.168.1.45"

# 1. re.search: Find first match anywhere in string
match = re.search(r"UserID:\s*(\d+)", log_line)
if match:
    print("Found User ID:", match.group(1))

# 2. re.findall: Extract all occurrences
ip_matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", log_line)
print("Extracted IPs: ", ip_matches)

# 3. re.sub: Anonymize sensitive numbers
masked_log = re.sub(r"UserID:\s*\d+", "UserID: [REDACTED]", log_line)
print("Masked Log:    ", masked_log)
```

#### Output:
```text
Found User ID: 8492
Extracted IPs:  ['192.168.1.45']
Masked Log:     2026-09-06 10:14:22 [ERROR] UserID: [REDACTED] failed login from IP: 192.168.1.45
```

---

## 4. Named Capture Groups & Lookarounds

```python
import re

text = "Revenue: $1,250.00 | Cost: $850.50 | Profit: $399.50"

# Named group + Positive Lookbehind (?<=\$): Match number preceded by dollar sign
pattern = r"\$(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2})"

for m in re.finditer(pattern, text):
    raw_str = m.group("amount").replace(",", "")
    print(f"Extracted Dollar Figure: ${float(raw_str):.2f}")
```

#### Output:
```text
Extracted Dollar Figure: $1250.00
Extracted Dollar Figure: $850.50
Extracted Dollar Figure: $399.50
```

---

## 5. JSON Parsing & Nested Extraction

```python
import json

raw_json_payload = """{
  "status": "success",
  "data": {
    "total_models": 2,
    "models": [
      {"name": "XGBoost", "accuracy": 0.942, "latency_ms": 12},
      {"name": "LightGBM", "accuracy": 0.948, "latency_ms": 9}
    ]
  }
}"""

parsed = json.loads(raw_json_payload)

# Extract nested properties
fastest_model = min(parsed["data"]["models"], key=lambda m: m["latency_ms"])
print(f"Top Model: {fastest_model['name']} with {fastest_model['latency_ms']}ms latency!")
```

#### Output:
```text
Top Model: LightGBM with 9ms latency!
```

---

## 6. RESTful Web APIs with Jittered Exponential Backoff

When consuming rate-limited REST APIs in production data pipelines:

```python
import time
import random

def mock_resilient_api_call(url: str, max_retries: int = 3):
    """Simulates API call with jittered exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            # Simulate network/rate-limit failure on first attempt
            if attempt == 1:
                raise ConnectionError("429 Too Many Requests")
            # Successful response
            return {"status": 200, "data": "Telemetry payload received"}
        except ConnectionError as err:
            if attempt == max_retries:
                raise
            # Backoff formula: 2^attempt + uniform jitter
            sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
            print(f"⚠️ Attempt {attempt} failed ({err}). Retrying in {sleep_time:.2f}s...")
            time.sleep(0.1)  # Compressed sleep for demonstration

response = mock_resilient_api_call("https://api.internal.ai/v1/metrics")
print("API Response:", response)
```

#### Output:
```text
⚠️ Attempt 1 failed (429 Too Many Requests). Retrying in 2.34s...
API Response: {'status': 200, 'data': 'Telemetry payload received'}
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Clean Email Extractor
**Task:** Given a raw text blurb with mixed characters and messy formatting, write a regex to extract all valid email addresses:
```python
blurb = "Reach out to admin@company.org or support-team@sub.domain.co.uk for inquiries. Avoid invalid@."
```

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import re

blurb = "Reach out to admin@company.org or support-team@sub.domain.co.uk for inquiries. Avoid invalid@."
email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

valid_emails = re.findall(email_pattern, blurb)
print("Extracted Valid Emails:\n", valid_emails)
```
#### Output:
```text
Extracted Valid Emails:
 ['admin@company.org', 'support-team@sub.domain.co.uk']
```
</details>

---

## 8. Quick Reference Cheat Sheet

| Operation | Regex Pattern / Python Code | Description |
|---|---|---|
| **Digits** | `r"\d+"` | 1 or more numbers |
| **Word Boundary**| `r"\bWORD\b"` | Matches whole word only |
| **Lookahead** | `r"foo(?=bar)"` | Matches "foo" only if followed by "bar" |
| **Lookbehind** | `r"(?<=\$)\d+"` | Matches digits preceded by dollar sign |
| **JSON Parse** | `json.loads(string)` | Converts JSON string to Python dictionary |
| **JSON Dump** | `json.dumps(obj, indent=2)` | Formats dictionary as clean JSON string |
