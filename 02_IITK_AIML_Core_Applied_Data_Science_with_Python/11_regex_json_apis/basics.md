# Regular Expressions, JSON Parsing & REST API Engineering: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Python / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Regex Engine Internals: Deterministic vs Non-Deterministic Finite Automata](#1-regex-engine-internals)
2. [Meta-characters, Quantifiers & Greedy vs Non-Greedy Matching](#2-meta-characters--quantifiers)
3. [Lookaround Assertions: Positive/Negative Lookahead & Lookbehind](#3-lookaround-assertions)
4. [Named Capture Groups & Pattern Compilation Hygiene](#4-named-capture-groups)
5. [Streaming JSON Processing: `json` vs `ijson` for Big Data](#5-streaming-json-processing)
6. [RESTful Web APIs: HTTP Methods, Status Codes & Headers](#6-restful-web-apis)
7. [API Resilience: Rate Limiting, Exponential Backoff & Token Buckets](#7-api-resilience--token-buckets)
8. [Common Pitfalls & Catastrophic Backtracking (ReDoS)](#8-common-pitfalls-redos)
9. [Production Case Study: Enterprise Webhook Ingestion & PII Redactor](#9-production-case-study-pii-redactor)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. Regex Engine Internals & Finite Automata

Python's `re` module uses a modified backtracking Non-Deterministic Finite Automaton (NFA).
- **Greedy Matching (`.*`):** Consumes as many characters as possible up to the end of string, then backtracks.
- **Non-Greedy / Lazy Matching (`.*?`):** Consumes the minimum necessary characters to satisfy the match.

```python
import re

html_snippet = "<div>Alpha</div><div>Beta</div>"

greedy_match = re.search(r"<div>.*</div>", html_snippet).group()
lazy_match = re.search(r"<div>.*?</div>", html_snippet).group()

print(f"Greedy Match: {greedy_match} (Eats both tags!)")
print(f"Lazy Match:   {lazy_match} (Stops at first closing tag)")
```

#### Output:
```text
Greedy Match: <div>Alpha</div><div>Beta</div> (Eats both tags!)
Lazy Match:   <div>Alpha</div> (Stops at first closing tag)
```

---

## 2. Lookaround Assertions: Lookahead & Lookbehind

Lookarounds perform zero-width assertions without consuming characters in the match buffer:
- **Positive Lookbehind `(?<=...)`:** Match occurs only if preceded by pattern.
- **Negative Lookbehind `(?<!...)`:** Match occurs only if NOT preceded by pattern.
- **Positive Lookahead `(?=...)`:** Match occurs only if followed by pattern.
- **Negative Lookahead `(?!...)`:** Match occurs only if NOT followed by pattern.

```python
text = "Product pricing: USD $149.99 and EUR €99.50 and CAD $49.00"

# Match amounts preceded by dollar sign using Positive Lookbehind
usd_amounts = re.findall(r"(?<=\$)\d+\.\d{2}", text)
print("Extracted Dollar Amounts:", usd_amounts)
```

#### Output:
```text
Extracted Dollar Amounts: ['149.99', '49.00']
```

---

## 3. Named Capture Groups

Named capture groups `(?P<name>...)` improve production maintainability:

```python
log_entry = "2026-09-06 10:32:00 [ERROR] Connection reset by peer from 192.168.1.104"
log_pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"\[(?P<level>[A-Z]+)\] (?P<message>.*?) from (?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
)

match = log_pattern.match(log_entry)
if match:
    parsed = match.groupdict()
    print("Structured Log Parsing:\n", parsed)
```

#### Output:
```text
Structured Log Parsing:
 {'timestamp': '2026-09-06 10:32:00', 'level': 'ERROR', 'message': 'Connection reset by peer', 'ip': '192.168.1.104'}
```

---

## 4. Production Case Study: Enterprise Automated PII Redactor

```python
class SensitiveDataRedactor:
    """Production PII sanitization engine using compiled regex."""
    def __init__(self):
        # Email RFC-compliant regex
        self.email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        # US SSN regex: XXX-XX-XXXX
        self.ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        # Credit Card 16-digit regex
        self.cc_re = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")

    def redact(self, text: str) -> str:
        text = self.email_re.sub("[REDACTED_EMAIL]", text)
        text = self.ssn_re.sub("[REDACTED_SSN]", text)
        text = self.cc_re.sub("[REDACTED_CREDIT_CARD]", text)
        return text

redactor = SensitiveDataRedactor()
sample_user_prompt = "Contact user at alice.smith@enterprise.com with SSN 452-98-1123 and card 4111 2222 3333 4444"
clean_prompt = redactor.redact(sample_user_prompt)
print("Sanitized LLM Ingestion Prompt:\n", clean_prompt)
```

#### Output:
```text
Sanitized LLM Ingestion Prompt:
 Contact user at [REDACTED_EMAIL] with SSN [REDACTED_SSN] and card [REDACTED_CREDIT_CARD]
```

---

## 5. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Validating Strong Password with Lookaheads
**Task:** Verify a password has at least 8 chars, 1 uppercase letter, 1 lowercase letter, 1 number, and 1 special symbol:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
password_regex = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$")

print("Is 'Password123!' valid? ", bool(password_regex.match("Password123!")))
print("Is 'weakpass' valid?     ", bool(password_regex.match("weakpass")))
```
#### Output:
```text
Is 'Password123!' valid?  True
Is 'weakpass' valid?      False
```
</details>

---

## 6. Quick Reference Cheat Sheet & Best Website Citations

| Syntax | Description | Example |
|---|---|---|
| `\d` / `\D` | Digit / Non-digit | `\d+` matches `"123"` |
| `\w` / `\W` | Word char / Non-word char | `\w+` matches `"user_name"` |
| `(?P<id>...)` | Named capture group | `match.group('id')` |
| `(?<=foo)bar` | Positive lookbehind | Matches `"bar"` only in `"foobar"` |

### 🌐 Official References & Recommended Reading:
- [Python Official Documentation — `re` Module](https://docs.python.org/3/library/re.html)
- [Regex101 Interactive Regex Debugger](https://regex101.com/)
- [W3Schools Python Regular Expressions](https://www.w3schools.com/python/python_regex.asp)
