# Chapter 11: Regular Expressions, JSON Streaming & Web APIs
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Modern data ingestion requires extracting unstructured textual signals via regular expressions, streaming semi-structured JSON payloads, and interfacing with distributed web microservices.

```
                  REST API EXTRACTION & BACKOFF PIPELINE
       Client Request ──► Send HTTPS GET / POST
                                │
                      Status 429 / 503?
                      ├── YES ──► Sleep = Base × (2 ^ attempt) + Jitter
                      │           Retry Request!
                      │
                      └── NO  ──► Parse JSON Payload via Streaming Parser
```

---

## 2. Deep Theoretical Foundations

### 1. Regular Expression Automata: DFA vs NFA
- **Deterministic Finite Automaton (DFA):** Processes each input character exactly once in $O(N)$ linear time. Does not support backreferences.
- **Non-Deterministic Finite Automaton (NFA - Python `re`):** Employs backtracking. While expressive, pathological nested patterns like `(a+)+$` evaluated on `aaaaX` lead to **catastrophic backtracking** with exponential $O(2^N)$ time complexity. Always anchor expressions and avoid ambiguous nested repetitions.

### 2. Distributed API Reliability & Jittered Exponential Backoff
When querying high-throughput endpoints, naive retries trigger thundering herd problems. Full jitter exponential backoff distributes retry load uniformly across client workers:
$$t_{\text{sleep}} = \text{Uniform}\left(0, \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{attempt}})\right)$$

---

## 3. Production Implementation: Resilient API Client with Streaming JSON

```python
import time
import random
import json
import urllib.request
import urllib.error
from typing import Any, Generator

class ResilientApiClient:
    """Production REST client featuring full-jitter exponential backoff."""
    def __init__(self, base_delay: float = 0.5, max_delay: float = 30.0, max_retries: int = 4):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries

    def fetch_with_retry(self, url: str) -> dict[str, Any]:
        attempt = 0
        while attempt <= self.max_retries:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'AIML-Production-Pipeline/1.0'})
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as err:
                if err.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    attempt += 1
                    # Full Jitter backoff formula
                    ceiling = min(self.max_delay, self.base_delay * (2 ** attempt))
                    sleep_time = random.uniform(0, ceiling)
                    time.sleep(sleep_time)
                else:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(random.uniform(0, min(self.max_delay, self.base_delay * (2 ** attempt))))
                else:
                    raise
        raise RuntimeError(f"Max retries exceeded fetching: {url}")
```
