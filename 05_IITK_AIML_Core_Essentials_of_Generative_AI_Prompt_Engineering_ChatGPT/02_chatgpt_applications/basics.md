# Enterprise LLM Applications, OpenAI API & Function Calling
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Chat Completion API Architecture](#1-the-chat-completion-api-architecture)
2. [Conversation History & Context Window Management](#2-conversation-history--context-window-management)
3. [Tool / Function Calling (Connecting LLMs to Databases & APIs)](#3-tool--function-calling)
4. [Structured Outputs with Strict JSON Schema](#4-structured-outputs-with-strict-json-schema)
5. [Token Economics & BPE Tokenization (Tiktoken)](#5-token-economics--bpe-tokenization)
6. [Real-Time Streaming Responses (Server-Sent Events)](#6-real-time-streaming-responses)
7. [Try It Yourself! (Hands-On Practice Exercises)](#7-try-it-yourself-hands-on-practice-exercises)
8. [Quick Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. The Chat Completion API Architecture

The modern LLM interaction loop passes a message sequence containing three primary roles:
- **`system`:** High-level instructions, constraints, and identity persona.
- **`user`:** The human's query or prompt.
- **`assistant`:** Model responses (used for multi-turn conversational history).

```
                 CHAT COMPLETION MESSAGE ARRAY DATAFLOW
    [ {"role": "system",    "content": "You are a financial advisor."}  ]
    [ {"role": "user",      "content": "Should I invest in Index Funds?"}]
    [ {"role": "assistant", "content": "Index funds provide diversification..."}]
    [ {"role": "user",      "content": "What is an expense ratio?"}     ] ◄── New question inherits context!
                  │
                  ▼ LLM Processing
    "An expense ratio is the annual percentage fee..."
```

---

## 2. Tool / Function Calling (Connecting LLMs to Code)

Function calling allows models like GPT-4 to output structured arguments calling your own backend functions:

```
                            FUNCTION CALLING LIFECYCLE
  1. User asks: "What's the weather in Seattle?"
         │
         ▼ 2. Model outputs JSON tool call:
  { "name": "get_weather", "arguments": "{\"location\": \"Seattle, WA\"}" }
         │
         ▼ 3. Python backend executes real database / API function:
  result = requests.get("https://weather.api?loc=Seattle").json()  --> "58°F, Rain"
         │
         ▼ 4. Pass tool result back to Model:
  [ {"role": "tool", "content": "58°F, Rain"} ]
         │
         ▼ 5. Model synthesizes natural response:
  "The current weather in Seattle is 58°F with light rain."
```

```python
import json

# Define tool schema adhering to JSON Schema standard
weather_tool = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Executes SQL query against enterprise warehouse.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "Valid SELECT SQL query string."
                }
            },
            "required": ["sql_query"]
        }
    }
}

print("Registered Tool Definition:\n", json.dumps(weather_tool, indent=2))
```

#### Output:
```text
Registered Tool Definition:
 {
  "type": "function",
  "function": {
    "name": "query_database",
    "description": "Executes SQL query against enterprise warehouse.",
    "parameters": {
      "type": "object",
      "properties": {
        "sql_query": {
          "type": "string",
          "description": "Valid SELECT SQL query string."
        }
      },
      "required": [
        "sql_query"
      ]
    }
  }
}
```

---

## 3. Token Economics & BPE Tokenization

LLMs do not read words; they process **tokens** generated via Byte Pair Encoding (BPE). In English, 1 token is roughly 4 characters or 0.75 words:

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
text = "Artificial intelligence and deep learning revolution."

tokens = encoding.encode(text)
token_words = [encoding.decode([t]) for t in tokens]

print(f"Original Text:   '{text}'")
print(f"Token IDs:       {tokens}")
print(f"Token Chunks:    {token_words}")
print(f"Total Tokens:    {len(tokens)} (Word count: {len(text.split())})")
```

#### Output:
```text
Original Text:   'Artificial intelligence and deep learning revolution.'
Token IDs:       [28399, 13783, 323, 3350, 4673, 8567, 13]
Token Chunks:    ['Artificial', ' intelligence', ' and', ' deep', ' learning', ' revolution', '.']
Total Tokens:    7 (Word count: 6)
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Sliding Window Context Trimmer
**Task:** Given a conversation history list of message dictionaries and a maximum token allowance of 100 tokens, write a function to retain the `system` message while trimming the oldest `user`/`assistant` messages from the history when tokens exceed the budget:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import tiktoken

def trim_history(messages, max_tokens=100):
    enc = tiktoken.get_encoding("cl100k_base")
    sys_msg = [m for m in messages if m['role'] == 'system']
    conv_msgs = [m for m in messages if m['role'] != 'system']
    
    # Calculate token count
    def count_tokens(msg_list):
        return sum(len(enc.encode(m['content'])) for m in msg_list)
    
    while count_tokens(sys_msg + conv_msgs) > max_tokens and len(conv_msgs) > 1:
        conv_msgs.pop(0)  # Evict oldest message
        
    return sys_msg + conv_msgs

sample_history = [
    {"role": "system", "content": "You are a customer assistant."},
    {"role": "user", "content": "What is refund policy? " * 5},
    {"role": "assistant", "content": "Refunds are processed in 14 days. " * 5},
    {"role": "user", "content": "Can I return open items?"}
]

trimmed = trim_history(sample_history, max_tokens=40)
print(f"Retained {len(trimmed)} of {len(sample_history)} messages.")
print("Active Messages:", [m['role'] for m in trimmed])
```
#### Output:
```text
Retained 2 of 4 messages.
Active Messages: ['system', 'user']
```
</details>

---

## 5. Quick Reference Cheat Sheet

| Feature | Key Parameter | Description |
|---|---|---|
| **Model** | `model="gpt-4o"` | Selects target LLM |
| **Tools** | `tools=[{"type": "function", ...}]` | Enables function calling capabilities |
| **Tool Choice** | `tool_choice="auto"` or `tool_choice="required"`| Forces tool invocation |
| **Response Format**| `response_format={"type": "json_object"}`| Guarantees valid JSON output |
| **Stream** | `stream=True` | Yields Server-Sent Events (SSE) token chunks |
