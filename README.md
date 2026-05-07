# GroundedSearch — Wikipedia RAG with Gemma 4 31B

A grounded AI search assistant that answers questions by iteratively searching Wikipedia, then synthesising a factual answer using **Gemma 4 31B** via Google AI Studio. Built with Flask and a custom dark-theme chat UI.

> Adapted from the [Anthropic Wikipedia Search Cookbook](https://github.com/anthropics/anthropic-cookbook) (originally written for Claude 2). Modernised with the `google-genai` SDK, a three-phase RAG pipeline, and real-time SSE streaming.

---

## Demo

```
User  →  "Which movie came out first: Oppenheimer, or Are You There God It's Me Margaret?"

App   →  🔍 Searching Wikipedia: "Oppenheimer film"
              ✓  Oppenheimer (film)
         🔍 Searching Wikipedia: "Are You There God Margaret film"
              ✓  Are You There God? It's Me, Margaret. (film)

         Are You There God? It's Me, Margaret. came out first,
         released on April 28, 2023. Oppenheimer followed on
         July 21, 2023.

         Sources: [Oppenheimer (film) ↗]  [Are You There God?... ↗]
```

---

## Why Wikipedia as a "Vector Database"?

Traditional RAG systems embed documents into a vector store and do semantic search. This project takes a different — and cost-free — approach:

| Approach | This project | Traditional RAG |
|---|---|---|
| Knowledge source | Wikipedia (free, always up-to-date) | Your own document store |
| Retrieval method | Keyword search via Wikipedia API | Semantic vector search |
| Embedding cost | Zero | Paid embedding API |
| Hallucination risk | Low — model answers from retrieved text | Low — same grounding principle |
| Setup complexity | Minimal | Requires vector DB + embedding pipeline |

The model is explicitly instructed **not to answer from memory** — it must retrieve Wikipedia pages first, then compose its answer from that content only.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Browser (Chat UI)                   │
│  Dark glassmorphism UI · Live search progress        │
│  Token streaming · Wikipedia source cards            │
└───────────────────┬──────────────────────────────────┘
                    │  POST /chat  (SSE stream)
┌───────────────────▼──────────────────────────────────┐
│              Flask  (groundedsearch.py)               │
│  Background thread + queue + 10s heartbeat           │
│  Prevents browser SSE timeout during API calls       │
└───────────────────┬──────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────┐
│           RAG Engine  (rag_engine.py)                 │
│                                                      │
│  Phase 1 — Retrieval loop                            │
│    Gemma generates <search_query> tags               │
│    Engine intercepts → calls Wikipedia               │
│    Injects <search_results> → repeats                │
│    (up to 5 iterations)                              │
│                                                      │
│  Phase 2 — Answer synthesis                          │
│    Extracted <information> fed to clean prompt       │
│    Gemma streams the final answer token-by-token     │
└───────────────┬──────────────────────────────────────┘
                │
┌───────────────▼──────────────┐  ┌────────────────────┐
│  Gemma 4 31B                 │  │  Wikipedia API      │
│  Google AI Studio (free tier)│  │  wikipedia library  │
│  google-genai SDK            │  │  + user-agent set   │
└──────────────────────────────┘  └────────────────────┘
```

### Three-phase pipeline (from the cookbook)

The pipeline preserves the original cookbook's core insight — **don't let the model answer while it searches**:

```
Phase 1a  →  Tool description prompt
              Gemma plans searches in <scratchpad>
              Emits <search_query>keywords</search_query>

Phase 1b  →  Engine intercepts the tag, queries Wikipedia
              Injects <search_results> into the prompt
              Gemma evaluates quality in <search_quality>
              Repeats until <information> tags are emitted

Phase 2   →  A fresh, clean prompt containing only the
              extracted <information> and the original question
              Gemma synthesises the final answer (streamed)
```

The separation of phases prevents the model from "pre-committing" to an answer while still gathering evidence — a failure mode explicitly described in the cookbook.

---

## Project Structure

```
WikiPedia/
├── groundedsearch.py          # Flask app — routes + SSE streaming
├── rag_engine.py              # RAG pipeline — Gemma + Wikipedia
├── templates/
│   └── index.html             # Chat UI — dark theme, streaming, source cards
├── .env                       # API key + model config (not committed)
├── requirements.txt           # Python dependencies
├── wikipedia-search-cookbook.ipynb   # Original Anthropic cookbook (reference)
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A free [Google AI Studio](https://aistudio.google.com) account

### 2. Get your API key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **Create API key**
3. Copy the key

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Open `.env` and paste your key:

```env
GEMINI_API_KEY=your_key_here
GEMMA_MODEL=gemma-4-31b-it
FLASK_SECRET_KEY=wikisearch_grounded_2024
```

> **Model ID note:** `gemma-4-31b-it` is the instruction-tuned variant of Gemma 4 31B. Verify the exact ID from your AI Studio dashboard if the app throws a model-not-found error.

### 5. Run

```bash
python groundedsearch.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## SSE Streaming Architecture

Standard Flask + SSE breaks on long AI inference calls because the browser will drop a silent connection after ~30 seconds. This project solves it with a **thread + queue + heartbeat** pattern:

```python
# groundedsearch.py

def run_engine():
    for event in engine.search(query):
        event_queue.put(event)        # RAG engine runs in background thread
    event_queue.put(None)             # sentinel — tells SSE loop to stop

threading.Thread(target=run_engine).start()

def generate():
    while True:
        try:
            event = event_queue.get(timeout=10)
            if event is None: break
            yield f"data: {json.dumps(event)}\n\n"
        except queue.Empty:
            yield ": heartbeat\n\n"   # SSE comment keeps connection alive
```

The `: heartbeat` line is an SSE comment — browsers ignore it but the TCP connection stays open, preventing timeout during long Gemma inference calls.

---

## SSE Event Reference

The `/chat` endpoint streams newline-delimited JSON events:

| Event type | Payload | UI effect |
|---|---|---|
| `search_start` | `{"query": "Oppenheimer film"}` | Amber spinner appears |
| `search_done` | `{"title": "...", "url": "..."}` | Spinner → green dot + Wikipedia link |
| `search_empty` | `{"query": "..."}` | Spinner → grey dot + "no results, retrying…" |
| `answer_start` | `{}` | Search section collapses, answer area opens |
| `token` | `{"content": "..."}` | Token appended and rendered as Markdown |
| `sources` | `{"sources": [...]}` | Source cards appear below answer |
| `done` | `{}` | Streaming cursor removed |
| `error` | `{"message": "..."}` | Red error banner shown |

---

## Performance Notes

### Why answers take 25–45 seconds

The latency is split across three types of work:

```
Wikipedia search + page fetch        ~1–3 s   per search
Gemma generation (retrieval phase)   ~3–8 s   per attempt  (300 token cap)
Gemma generation (answer phase)      ~10–20 s              (1024 token cap, streamed)
```

For a two-search comparison query: `3 + 3 + 8 + 3 + 8 + 15 ≈ 40 seconds` total.

### Why max_output_tokens=300 in the retrieval loop

Generation speed on the free tier is approximately 15–40 tokens/second. With the previous limit of 1024 tokens, if Gemma wrote a verbose `<search_quality>` section (400–800 tokens) before the next `<search_query>` tag, that produced **10–53 seconds of silence** — the browser appeared frozen.

Capping at 300 tokens limits any single retrieval generation to **7–20 seconds maximum**, regardless of how verbose the model tries to be.

### Why 20,000 character page truncation

Gemma 4 31B has a 256k token context window with fast parallel prefill. 20,000 characters (~5,000 tokens) is approximately 2% of the context window. Prefill of this content takes roughly 2–5 seconds — the content size is not the bottleneck.

---

## Free Tier Limits (Google AI Studio)

| Resource | Limit |
|---|---|
| Requests per minute (RPM) | 15 |
| Tokens per minute (TPM) | Unlimited |
| Requests per day (RPD) | 1,500 |

A single complex query uses 2–4 API requests (retrieval attempts + answer). At 1,500 RPD you can handle ~375–750 queries per day on the free tier.

---

## Differences from the Original Cookbook

| Aspect | Cookbook (Claude 2) | This project (Gemma 4 31B) |
|---|---|---|
| LLM SDK | `anthropic` (legacy) | `google-genai` |
| Model | `claude-2` | `gemma-4-31b-it` |
| Tokenizer | Anthropic tokenizer for truncation | Character-based truncation |
| Stop sequences | `AI_PROMPT` / `HUMAN_PROMPT` format | `</search_query>` via `GenerateContentConfig` |
| Serving | Jupyter notebook only | Flask app with SSE streaming |
| Frontend | None (stdout) | Dark chat UI with live search progress |
| Concurrency | Synchronous | Background thread + queue |
| Error handling | Basic | Search failures, rate limits, safety filters |

---

## Troubleshooting

**`model not found` error**
Check the exact model ID in your AI Studio dashboard and update `GEMMA_MODEL` in `.env`.

**`JSONDecodeError` from Wikipedia**
Wikipedia's API occasionally returns empty responses when rate-limited. The engine retries automatically and logs `[wiki] search() failed`. Wait a few seconds and try again.

**"Network error" in the browser**
Usually means the Flask server crashed. Check the terminal for a Python traceback.

**Spinner stuck / no answer after search**
Restart the Flask server — the SSE connection may have been left in a broken state from a previous failed request.

**Answer quality is poor**
Wikipedia's keyword search sometimes returns tangentially related articles. Try rephrasing your question to use proper nouns the model can turn into clean search keywords (e.g. "Oppenheimer 2023 film" instead of "the movie with the bomb").

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemma 4 31B (Google AI Studio, free tier) |
| LLM SDK | `google-genai` |
| Knowledge base | Wikipedia (`wikipedia` Python library) |
| Backend | Flask 3 |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Vanilla HTML/CSS/JS, Inter font, Marked.js |
| Config | `python-dotenv` |
