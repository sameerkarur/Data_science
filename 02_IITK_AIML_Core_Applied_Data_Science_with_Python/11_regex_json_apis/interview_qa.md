# Interview Q&A — Regular Expressions, JSON & REST APIs

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain greedy vs non-greedy (lazy) quantifiers in regular expressions.

**Answer:** Greedy quantifiers ('*', '+', '{m,n}') match as many characters as possible while still allowing the overall pattern to match. Adding a '?' makes them lazy/non-greedy ('*?', '+?'), matching the absolute minimum number of characters required to satisfy the expression. For example, matching '<.*>' on '<p>hello</p>' matches the full string greedily, while '<.*?>' matches just '<p>'.

### Q2. What are regex Lookahead and Lookbehind assertions (Zero-Width Assertions)?

**Answer:** Zero-width assertions match a position in text based on what follows or precedes it without consuming characters in the match result. Positive lookahead '(?=pattern)', negative lookahead '(?!pattern)', positive lookbehind '(?<=pattern)', and negative lookbehind '(?<!pattern)'. Example: '(?<=\$)\d+' extracts digits preceded by a dollar sign without including the '$'.

### Q3. Explain the difference between 're.match()', 're.search()', and 're.findall()'.

**Answer:** 're.match()' checks for a match ONLY at the very beginning of the string. 're.search()' scans the entire string and returns the first Match object found anywhere. 're.findall()' scans the entire string and returns all non-overlapping matches as a list of strings or tuples.

### Q4. What is the purpose of 're.compile()' and when should it be used?

**Answer:** 're.compile(pattern)' compiles a regex string into a reusable RegexObject bytecode in memory. It improves performance when the same regex is executed repeatedly across thousands or millions of text records, avoiding re-parsing and compiling overhead on every iteration.

### Q5. How do capturing groups '(...)' differ from non-capturing groups '(?:...)'?

**Answer:** Capturing groups '(...)' extract the matched substring, storing it in match groups accessible via 'match.group(1)'. Non-capturing groups '(?:...)' group expressions logically (for quantifiers or alternation) without allocating memory to capture or store the substring, improving execution speed.

### Q6. What are Named Capturing Groups and how do you access them?

**Answer:** Named capturing groups use the syntax '(?P<name>pattern)'. Extracted values can be accessed by name: 'match.group('name')' or extracted into a dictionary via 'match.groupdict()', making regex parsing code self-documenting and resilient to group reordering.

### Q7. Explain regex catastrophic backtracking and how to prevent it.

**Answer:** Catastrophic backtracking occurs when nested ambiguous quantifiers (e.g. '(a+)+$') evaluate against non-matching strings (e.g. 'aaaaaX'). The engine explores an exponential number of combinatorial paths (O(2^N)), freezing the CPU (ReDoS attack). Prevent by using atomic grouping, possessive quantifiers, or avoiding nested variable-length quantifiers.

### Q8. What is RESTful API architecture and what are its core architectural constraints?

**Answer:** REST (Representational State Transfer) constraints: (1) Client-Server separation, (2) Statelessness (no client session state stored on server), (3) Cacheability, (4) Uniform Interface (resource identification via URIs, self-descriptive messages), (5) Layered System (proxies, load balancers), and (6) Code-on-Demand (optional).

### Q9. Explain the standard HTTP request methods and their idempotency.

**Answer:** GET: retrieves a resource (safe, idempotent). POST: creates a new subordinate resource (neither safe nor idempotent). PUT: creates or completely replaces a resource (idempotent). PATCH: partially updates a resource (non-idempotent by default). DELETE: removes a resource (idempotent). Idempotent means multiple identical requests produce the same server state as a single request.

### Q10. Differentiate between HTTP status code ranges: 2xx, 3xx, 4xx, and 5xx.

**Answer:** 2xx: Success (200 OK, 201 Created, 204 No Content). 3xx: Redirection (301 Moved Permanently, 304 Not Modified). 4xx: Client Error (400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Too Many Requests). 5xx: Server Error (500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable).

### Q11. What is the difference between 401 Unauthorized and 403 Forbidden?

**Answer:** 401 Unauthorized means authentication is required and has failed or has not yet been provided (missing/invalid credentials/token). 403 Forbidden means the server understands the client's identity (authenticated), but the client lacks permission/authorization to access the requested resource.

### Q12. How do API Rate Limiting algorithms work (Token Bucket vs Leaky Bucket)?

**Answer:** Token Bucket: tokens are added to a bucket at a constant rate up to capacity; each request consumes a token, allowing bursts of requests up to bucket capacity. Leaky Bucket: requests enter a FIFO queue and leak out at a strictly constant rate, smoothing traffic and enforcing rigid throughput limits.

### Q13. What is Exponential Backoff with Jitter in API client pipelines?

**Answer:** When an API returns 429 (Rate Limit) or 503 (Unavailable), exponential backoff doubles the wait time between successive retry attempts (e.g. 1s, 2s, 4s, 8s). Adding random 'jitter' (random noise to wait time) desynchronizes retries from concurrent client threads, preventing the 'Thundering Herd' problem on the server.

### Q14. Explain OAuth 2.0 authorization framework and Bearer Tokens.

**Answer:** OAuth 2.0 delegates user authorization to a third-party service without sharing user passwords. An authorization server issues a short-lived Bearer Token (access token). The client sends this token in the HTTP Authorization header ('Authorization: Bearer <token>') to access protected API endpoints.

### Q15. What is JWT (JSON Web Token) and what are its three components?

**Answer:** JWT is a compact, URL-safe, digitally signed token consisting of three Base64URL-encoded parts separated by dots: Header (algorithm & token type), Payload (claims: user id, roles, expiration time 'exp'), and Signature (cryptographic HMAC-SHA256 or RSA signature verifying payload integrity).

### Q16. Explain the difference between 'json.loads()' and 'json.load()'.

**Answer:** 'json.loads(str)' parses a JSON formatted string into a Python dictionary/list. 'json.load(fp)' reads from a file pointer or stream directly, parsing the stream without loading the entire raw file into memory as a separate string.

### Q17. What causes a JSONDecodeError and how do you handle it defensively?

**Answer:** JSONDecodeError occurs when parsing malformed JSON (trailing commas, single quotes instead of double quotes, unquoted keys, truncated streams). Handle via 'try-except json.JSONDecodeError' and sanitize using regex or libraries like 'json5'.

### Q18. How does pagination work in REST APIs (Offset-based vs Cursor-based)?

**Answer:** Offset-based ('?page=2&limit=50') uses SQL OFFSET; it is simple but suffers from performance degradation on large tables (O(N) database scans) and produces duplicate or skipped items if rows are inserted between page queries. Cursor-based ('?cursor=eyJpZCI6MTB9') uses indexed sequential IDs, guaranteeing O(1) database lookups and stable pagination across concurrent writes.

### Q19. What are Webhooks and how do they differ from polling APIs?

**Answer:** Polling repeatedly queries an API at regular intervals to check for updates, wasting bandwidth and incurring latency. Webhooks are event-driven HTTP push notifications: the server sends an asynchronous HTTP POST request to the client's designated webhook URL the moment an event occurs.

### Q20. Explain GraphQL and how it solves Over-fetching and Under-fetching.

**Answer:** REST APIs often return fixed payloads with redundant fields (over-fetching) or require multiple chained API calls to assemble related resources (under-fetching). GraphQL provides a single endpoint where clients submit structured queries requesting the exact fields and nested relationships needed, returned in a single predictable response.

### Q21. How do you stream large JSON responses from an API without exhausting memory in Python?

**Answer:** Use the 'requests' library with 'stream=True': 'response = requests.get(url, stream=True)'. Process chunks iteratively using 'response.iter_lines()' or use 'ijson' (iterative JSON parser) to parse nested JSON elements as a stream of Python objects without loading the multi-gigabyte file into RAM.

### Q22. What is CORS (Cross-Origin Resource Sharing) and why does it exist?

**Answer:** CORS is a browser security mechanism that restricts web applications running at one origin from requesting resources from a different origin. Servers declare allowed origins, methods, and headers via HTTP response headers ('Access-Control-Allow-Origin: *') to prevent malicious cross-site data theft.

### Q23. What is API Idempotency-Key and why is it essential for payment and mutation APIs?

**Answer:** An Idempotency-Key is a unique client-generated UUID sent in headers ('Idempotency-Key: uuid'). If a network timeout occurs and the client retries the request, the server detects the key, skips re-executing the financial charge or mutation, and returns the cached response of the original operation.

### Q24. How do you handle API authentication using session cookies vs API tokens?

**Answer:** Session cookies are stored automatically by browsers and sent on every request; vulnerable to Cross-Site Request Forgery (CSRF). API tokens (Bearer/JWT) are stored in client application state and sent explicitly in Authorization headers, immune to CSRF and suitable for stateless mobile and microservice architectures.

### Q25. What is the difference between synchronous and asynchronous HTTP requests in Python?

**Answer:** Synchronous libraries ('requests', 'urllib') block the execution thread until the server responds, limiting throughput to 1 request at a time per thread. Asynchronous libraries ('aiohttp', 'httpx') use Python's 'asyncio' event loop to dispatch thousands of concurrent non-blocking HTTP requests across a single thread.

### Q26. Explain how to validate API request and response schemas using Pydantic.

**Answer:** Pydantic models define expected fields, types, and constraints via Python type annotations. Calling 'MyModel.model_validate(json_data)' parses the input, coerces compatible types, and raises structured 'ValidationError' exceptions if required fields are missing or types are violated.

### Q27. What are query parameters vs path parameters in REST URLs?

**Answer:** Path parameters identify a specific hierarchical resource: '/users/{user_id}/orders/{order_id}'. Query parameters filter, sort, or paginate that resource collection: '/users?role=admin&sort=created_at&limit=10'.

### Q28. What is OpenAPI (Swagger) specification?

**Answer:** OpenAPI is a machine-readable JSON/YAML standard for describing RESTful APIs. It documents endpoints, request parameters, response schemas, and authentication methods, enabling automatic interactive documentation generation, client SDK generation, and mock servers.

### Q29. How do you sanitize text before passing it to Regex engines to prevent ReDoS?

**Answer:** Use 're.escape(user_string)' to escape all special regex metacharacters (*, +, ?, ^, $, (, ), [, ], {, }, |, \) when incorporating untrusted user input into regex patterns, preventing pattern injection and catastrophic backtracking attacks.

### Q30. What is gRPC and how does it compare to REST/JSON APIs?

**Answer:** gRPC is a high-performance RPC framework developed by Google. It uses Protocol Buffers (compact binary serialization) over HTTP/2 (multiplexed streaming, header compression) instead of plain-text JSON over HTTP/1.1. It is 5x–10x faster and strictly typed, making it standard for internal microservices.
