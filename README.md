# GroundedSearch — Wikipedia RAG with Gemma 4 31B

A grounded AI search assistant that answers questions by iteratively searching Wikipedia, then synthesising a factual answer using **Gemma 4 31B** via Google AI Studio. Built with Flask and a custom dark-theme chat UI.

> Adapted from the [Anthropic Wikipedia Search Cookbook](https://github.com/anthropics/anthropic-cookbook) (originally written for Claude 2). Modernised with the `google-genai` SDK, a three-phase RAG pipeline, and real-time SSE streaming.

---

## Demo

**GroundedSearch mode** (thorough, ~25–45s):
```
User  →  "Which movie came out first: Oppenheimer, or Are You There God It's Me Margaret?"

App   →  🔍 Searching Wikipedia: "Oppenheimer movie"
              ✓  Oppenheimer (film)
         🔍 Searching Wikipedia: "Are You There God Margaret movie"
              ✓  Are You There God? It's Me, Margaret. (film)

         Are You There God? It's Me, Margaret. came out first,
         released on April 28, 2023. Oppenheimer followed on
         July 21, 2023.

         Sources: [Oppenheimer (film) ↗]  [Are You There God?... ↗]
```

**Fast mode** (quick, ~10–15s):
```
User  →  "Which movie came out first: Oppenheimer, or Are You There God It's Me Margaret?"

App   →  🔍 Searching Wikipedia: "Oppenheimer film"      ← parallel
         🔍 Searching Wikipedia: "Are You There God Margaret film"  ← parallel

         Are You There God? It's Me, Margaret. came out first
         (April 28, 2023). Oppenheimer released July 21, 2023.

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

## Two Modes

### GroundedSearch (thorough)

The full three-phase RAG pipeline from the original cookbook:

```
Phase 1a  →  Tool description prompt
              Gemma plans searches in <scratchpad>
              Emits <search_query>keywords</search_query>

Phase 1b  →  Engine intercepts the tag, queries Wikipedia
              Injects <search_results> into the prompt
              Gemma evaluates quality in <search_quality>
              Repeats until <information> tags are emitted (up to 5 iterations)

Phase 2   →  A fresh, clean prompt with only the extracted <information>
              Gemma synthesises the final answer (streamed)
```

Best for: multi-part questions, comparisons, research topics that need several Wikipedia sources.

### Fast mode (quick)

A simplified single-pass pipeline:

```
Step 1  →  Keyword extraction call (max 50 tokens, ~1s)
            Gemma extracts 1–2 search terms with disambiguation
            (e.g. "Oppenheimer film", "Are You There God Margaret film")

Step 2  →  Up to 2 Wikipedia searches run IN PARALLEL
            Both fetches fire simultaneously — same wall-clock time as one

Step 3  →  Single streaming answer call grounded in the combined Wikipedia context
```

Best for: factual lookups, single-subject questions, comparison of two things.

**Why parallel searches keep Fast mode fast:**
If a question involves two subjects, both Wikipedia fetches fire at the same time via `ThreadPoolExecutor`. Two parallel fetches at 2s each still take ~2s total — not 4s. You only pay for the slower of the two.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Browser (Chat UI)                   │
│  Dark glassmorphism UI · Mode toggle (Fast/Grounded) │
│  Per-mode chat history · New Chat button             │
│  Live search progress · Token streaming              │
│  Wikipedia source cards                              │
└───────────────────┬──────────────────────────────────┘
                    │  POST /chat  (SSE stream)
┌───────────────────▼──────────────────────────────────┐
│              Flask  (groundedsearch.py)               │
│  Background thread + queue + 10s heartbeat           │
│  Routes to fast_search() or search() based on mode   │
│  Prevents browser SSE timeout during API calls       │
└───────────────────┬──────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────┐
│           RAG Engine  (rag_engine.py)                 │
│                                                      │
│  _generate() retry helper                            │
│    Wraps all non-streaming API calls                 │
│    Auto-retries on 500/503 with exponential backoff  │
│                                                      │
│  search()  — GroundedSearch mode                     │
│    Phase 1: iterative <search_query> loop            │
│    Phase 2: answer synthesis                         │
│                                                      │
│  fast_search()  — Fast mode                          │
│    Keyword extraction → parallel Wikipedia fetches   │
│    → single streaming answer                         │
└───────────────┬──────────────────────────────────────┘
                │
┌───────────────▼──────────────┐  ┌────────────────────┐
│  Gemma 4 31B                 │  │  Wikipedia API      │
│  Google AI Studio (free tier)│  │  wikipedia library  │
│  google-genai SDK            │  │  2 results per query│
└──────────────────────────────┘  └────────────────────┘
```

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
| `search_start` | `{"query": "Oppenheimer film"}` | Amber pulsing dot appears |
| `search_done` | `{"query": "...", "title": "...", "url": "..."}` | Dot turns green + Wikipedia link |
| `search_empty` | `{"query": "..."}` | Dot turns grey + "no results, retrying…" |
| `answer_start` | `{}` | Search section collapses, answer area opens |
| `token` | `{"content": "..."}` | Token appended and rendered as Markdown |
| `sources` | `{"sources": [...]}` | Wikipedia source cards appear below answer |
| `done` | `{}` | Streaming cursor removed |
| `error` | `{"message": "..."}` | Spinner stopped, red error banner shown |

> **Note:** `search_done` now always includes the `query` field so the frontend can correctly resolve each search item independently — this is required for Fast mode's parallel searches where multiple `search_done` events arrive out of order.

---

## UI Features

### Mode toggle
A pill toggle in the header switches between **⚡ Fast** and **🔬 GroundedSearch**. Each mode maintains its own separate chat history — switching modes preserves conversations and switching back restores them. The toggle is blocked while a response is streaming.

### Per-mode chat history
Fast and GroundedSearch run as independent conversations. Switching from Fast to GroundedSearch shows that mode's previous messages, not a blank screen. **New Chat** only clears the active mode's history — the other mode is untouched.

### New Chat button
Clears only the current mode's conversation and restores the welcome screen with example prompts.

### Mode pill on responses
Each AI response shows a coloured pill (yellow ⚡ for Fast, purple 🔬 for GroundedSearch) so you can tell at a glance which mode produced which answer when comparing results.

---

## Performance Notes

### GroundedSearch: why answers take 25–45 seconds

```
Wikipedia search + page fetch        ~1–3 s   per search
Gemma generation (retrieval phase)   ~3–8 s   per attempt  (300 token cap)
Gemma generation (answer phase)      ~10–20 s              (1024 token cap, streamed)
```

For a two-search comparison query: `3 + 3 + 8 + 3 + 8 + 15 ≈ 40 seconds` total.

### Fast mode: why it stays fast with two searches

Fast mode uses `ThreadPoolExecutor` to run both Wikipedia fetches simultaneously. Two parallel fetches at ~2s each still complete in ~2s — not 4s. The only added cost over a single-subject question is a slightly larger answer prompt.

### Why max_output_tokens=300 in the GroundedSearch retrieval loop

Generation speed on the free tier is approximately 15–40 tokens/second. Without this cap, if Gemma wrote a verbose `<search_quality>` section (400–800 tokens) before the next `<search_query>` tag, that produced **10–53 seconds of silence** with the browser appearing frozen. Capping at 300 tokens limits any single retrieval generation to **7–20 seconds maximum**.

### Why 20,000 character page truncation

Gemma 4 31B has a 256k token context window with fast parallel prefill. 20,000 characters (~5,000 tokens) is approximately 2% of the context window — prefill is not the bottleneck. In Fast mode, when two articles are fetched, each is capped at 10,000 characters so the combined answer prompt stays under 20,000 characters total, preventing silent API failures.

### API retry logic

All non-streaming Gemma calls go through `_generate()`, a retry wrapper that automatically retries up to 2 times (with 1s and 2s backoff) on transient 500/503 errors from Google's free tier. Streaming answer calls also retry up to 3 times on the same conditions.

---

## Free Tier Limits (Google AI Studio)

| Resource | Limit |
|---|---|
| Requests per minute (RPM) | 15 |
| Tokens per minute (TPM) | Unlimited |
| Requests per day (RPD) | 1,500 |

A single GroundedSearch query uses 3–6 API requests (retrieval attempts + answer). Fast mode uses 2 requests (keyword extraction + answer). At 1,500 RPD you can handle ~250–750 queries per day on the free tier.

---

## Differences from the Original Cookbook

| Aspect | Cookbook (Claude 2) | This project (Gemma 4 31B) |
|---|---|---|
| LLM SDK | `anthropic` (legacy) | `google-genai` |
| Model | `claude-2` | `gemma-4-31b-it` |
| Tokenizer | Anthropic tokenizer for truncation | Character-based truncation |
| Stop sequences | `AI_PROMPT` / `HUMAN_PROMPT` format | `</search_query>` via `GenerateContentConfig` |
| Serving | Jupyter notebook only | Flask app with SSE streaming |
| Frontend | None (stdout) | Dark glassmorphism UI with live search progress |
| Concurrency | Synchronous | Background thread + queue + heartbeat |
| Search modes | Single mode | Fast (parallel) + GroundedSearch (iterative) |
| Chat history | N/A | Per-mode independent conversation history |
| Error handling | Basic | Retry logic, spinner cleanup, friendly messages |

---

## Troubleshooting

**`model not found` error**
Check the exact model ID in your AI Studio dashboard and update `GEMMA_MODEL` in `.env`.

**`JSONDecodeError` from Wikipedia**
Wikipedia's API occasionally returns empty responses when rate-limited. The engine retries automatically and logs `[wiki] search() failed`. Wait a few seconds and try again.

**"Network error" in the browser**
Usually means the Flask server crashed. Check the terminal for a Python traceback.

**`ServerError: 500 INTERNAL` from Google API**
These are transient free-tier errors. The retry logic handles them automatically (you'll see `[engine] API error, retrying` in the terminal). If they happen consistently, wait a minute — you may have hit the 15 RPM limit.

**Answer is correct but search dot stays amber/blinking**
This was a known bug (now fixed) where parallel `search_done` events were matched to the wrong UI element. If you still see it, hard-refresh the browser to clear cached JS.

**Answer quality is poor in Fast mode**
Wikipedia's keyword search sometimes returns off-topic articles for unusual titles. Switch to GroundedSearch mode — its iterative loop is better at self-correcting when the first search result is wrong.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemma 4 31B (Google AI Studio, free tier) |
| LLM SDK | `google-genai` |
| Knowledge base | Wikipedia (`wikipedia` Python library) |
| Backend | Flask 3 |
| Streaming | Server-Sent Events (SSE) |
| Concurrency | `threading` (SSE) + `concurrent.futures` (parallel wiki fetches) |
| Frontend | Vanilla HTML/CSS/JS, Inter font, Marked.js |
| Config | `python-dotenv` |
