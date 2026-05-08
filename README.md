# GroundedSearch — Wikipedia RAG with Gemma 4 31B

> **Educational Project:** This is built for learning purposes. Expect some lag in responses (10–45 seconds depending on mode). If you get an error, simply re-ask the same question, transient API errors are common on the free tier and usually resolve on the next attempt. If the answer seems incomplete or off-topic, switch modes and try again.

A grounded AI search assistant that answers questions by iteratively searching Wikipedia, then synthesising a factual answer using **Gemma 4 31B** via Google AI Studio. Built with Flask and a custom dark-theme chat UI.

> Adapted from the [Anthropic Wikipedia Search Cookbook](https://github.com/anthropics/anthropic-cookbook) (originally written for Claude 2). Modernised with the `google-genai` SDK, a three-phase RAG pipeline, and real-time SSE streaming.

**[Live Demo](https://groundedsearch.onrender.com):** Try it now! Deployed on Render free tier.

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
User  →  "Who is the most recently appointed Supreme Court justice in USA and India?"

App   →  🔍 Searching: "Chief Justice of the United States"   ← parallel
         🔍 Searching: "Chief Justice of India"               ← parallel

         The most recently confirmed US Supreme Court Justice is ...
         The current Chief Justice of India is Surya Kant, appointed November 24, 2025.

         Sources: [Chief Justice of the United States ↗]  [Chief Justice of India ↗]
```

---

## Tips for Best Results

| Situation                                             | What to do                                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Got a red error banner                                | Re-ask the same question — transient API 500s resolve on retry                                  |
| Answer is vague or says "context does not contain..." | Switch to**🔬 GroundedSearch** — it does iterative searches and finds more specific articles    |
| GroundedSearch is taking too long                     | Switch to**⚡ Fast** for a quicker answer                                                        |
| Search found the wrong Wikipedia article              | Rephrase with more specific proper nouns (e.g. "Oppenheimer 2023 film" instead of "Oppenheimer") |
| Spinner stuck after an error                          | The spinner now auto-stops on errors — if still stuck, refresh the page                         |

---

## Why Wikipedia as a "Vector Database"?

Traditional RAG systems embed documents into a vector store and do semantic search. This project takes a different — and cost-free — approach:

| Approach           | This project                             | Traditional RAG                         |
| ------------------ | ---------------------------------------- | --------------------------------------- |
| Knowledge source   | Wikipedia (free, always up-to-date)      | Your own document store                 |
| Retrieval method   | Keyword search via Wikipedia API         | Semantic vector search                  |
| Embedding cost     | Zero                                     | Paid embedding API                      |
| Hallucination risk | Low — model answers from retrieved text | Low — same grounding principle         |
| Setup complexity   | Minimal                                  | Requires vector DB + embedding pipeline |

The model is explicitly instructed **not to answer from memory** — it must retrieve Wikipedia pages first, then compose its answer from that content only.

---

## Two Modes

### ⚡ Fast mode (~10–15s)

A simplified single-pass pipeline designed for speed:

```
Step 1  →  Keyword extraction (max 50 tokens, ~1s)
            Gemma extracts 1–2 smart search terms with disambiguation rules:
            • Movies/books/songs → adds "film" / "novel" / "song"
            • Person queries (who is, most recent, appointed) → searches the
              ROLE article, not the institution
              e.g. "Chief Justice of India" not "Supreme Court of India"
            • Comparison questions → 2 lines, one term per subject

Step 2  →  Up to 2 Wikipedia searches IN PARALLEL
            Both fetches fire simultaneously — same wall-clock time as one

Step 3  →  Single streaming answer call grounded in the combined Wikipedia context
```

**When Fast mode may not give a perfect answer:**
Wikipedia search is keyword-based, not semantic. If the extracted keywords land on the wrong article (e.g. an institution overview instead of a person-role article), the model will correctly say "the context does not contain this information" rather than hallucinate. In that case, switch to GroundedSearch — its iterative loop self-corrects when the first result is wrong.

### 🔬 GroundedSearch mode (~25–45s)

The full three-phase RAG pipeline from the original cookbook:

```
Phase 1a  →  Tool description prompt
              Gemma plans searches in <scratchpad>
              Emits <search_query>keywords</search_query>  (1–4 words, atomic)

Phase 1b  →  Engine intercepts the tag, queries Wikipedia
              Injects <search_results> into the prompt
              Gemma evaluates quality in <search_quality>
              Repeats until <information> tags are emitted (up to 5 iterations)
              Between searches: UI shows "Analysing results, searching more…"

Phase 2   →  A fresh, clean prompt with only the extracted <information>
              Gemma synthesises the final answer (streamed token by token)
```

Best for: multi-part questions, comparisons, person/role lookups, anything that needs several Wikipedia sources.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Browser (Chat UI)                   │
│  Dark glassmorphism UI · Mode toggle (Fast/Grounded) │
│  Per-mode chat history · New Chat button             │
│  Live search progress · "Analysing…" between searches│
│  Token streaming · Wikipedia source cards            │
└───────────────────┬──────────────────────────────────┘
                    │  POST /chat  (SSE stream)
┌───────────────────▼──────────────────────────────────┐
│              Flask  (groundedsearch.py)               │
│  Background thread + queue + 10s heartbeat           │
│  Routes to fast_search() or search() based on mode   │
│  Prevents browser SSE timeout during long API calls  │
└───────────────────┬──────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────┐
│           RAG Engine  (rag_engine.py)                 │
│                                                      │
│  _generate() — safe API wrapper                      │
│    • 2-minute thread-based timeout (prevents hangs)  │
│    • Auto-retry on 500/503 with 1s/2s backoff        │
│    • Timeouts are not retried (hang won't un-hang)   │
│                                                      │
│  search()  — GroundedSearch mode                     │
│    Phase 1: iterative <search_query> loop            │
│    Emits thinking event between attempts             │
│    Phase 2: answer synthesis                         │
│                                                      │
│  fast_search()  — Fast mode                          │
│    Smart keyword extraction with disambiguation      │
│    → parallel Wikipedia fetches (ThreadPoolExecutor) │
│    → single streaming answer                         │
└───────────────┬──────────────────────────────────────┘
                │
┌───────────────▼──────────────┐  ┌────────────────────┐
│  Gemma 4 31B                 │  │  Wikipedia API      │
│  Google AI Studio (free tier)│  │  wikipedia library  │
│  google-genai SDK            │  │  2 results / query  │
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
├── Procfile                   # Render deployment start command
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

## Deployment (Render — free tier)

### Steps

1. Push to GitHub (`.env` is gitignored — never committed)
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Set build/start commands:

| Field         | Value                                                                          |
| ------------- | ------------------------------------------------------------------------------ |
| Build Command | `pip install -r requirements.txt`                                            |
| Start Command | `gunicorn groundedsearch:app --timeout 120 --workers 2 --bind 0.0.0.0:$PORT` |
| Instance Type | Free                                                                           |

5. Add environment variables in the Render dashboard:

| Key                | Value              |
| ------------------ | ------------------ |
| `GEMINI_API_KEY` | your actual key    |
| `GEMMA_MODEL`    | `gemma-4-31b-it` |

### Free tier behaviour

| Thing            | Limit / behaviour                                             |
| ---------------- | ------------------------------------------------------------- |
| Hours/month      | 750 (enough for 1 always-on service)                          |
| Inactivity sleep | Spins down after 15 min idle — ~30s cold start on next visit |
| Bandwidth        | 100 GB/month                                                  |
| Custom domain    | Free (add a CNAME)                                            |

---

## SSE Streaming Architecture

Standard Flask + SSE breaks on long AI inference calls because the browser drops a silent connection after ~30 seconds. This project solves it with a **thread + queue + heartbeat** pattern:

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
            yield ": heartbeat\n\n"   # SSE comment — keeps TCP connection open
```

The `: heartbeat` line is an SSE comment — browsers ignore it but the TCP connection stays open, preventing timeout during long Gemma inference calls.

---

## SSE Event Reference

The `/chat` endpoint streams newline-delimited JSON events:

| Event type       | Payload                                            | UI effect                                               |
| ---------------- | -------------------------------------------------- | ------------------------------------------------------- |
| `search_start` | `{"query": "Chief Justice of India"}`            | Amber pulsing dot appears                               |
| `search_done`  | `{"query": "...", "title": "...", "url": "..."}` | Dot turns green + Wikipedia link                        |
| `search_empty` | `{"query": "..."}`                               | Dot turns grey + "no results, retrying…"               |
| `thinking`     | `{"attempt": 2}`                                 | Header updates to "Analysing results, searching more…" |
| `answer_start` | `{}`                                             | Search section collapses, answer area opens             |
| `token`        | `{"content": "..."}`                             | Token appended and rendered as Markdown                 |
| `sources`      | `{"sources": [...]}`                             | Wikipedia source cards appear below answer              |
| `done`         | `{}`                                             | Streaming cursor removed                                |
| `error`        | `{"message": "..."}`                             | Spinner stopped, red "Search failed" banner             |

> `search_done` always includes `query` so the frontend resolves each search dot independently — required for Fast mode's parallel searches where two `search_done` events arrive simultaneously.

---

## UI Features

### Mode toggle

A pill toggle in the header switches between **⚡ Fast** and **🔬 GroundedSearch**. Toggling is blocked while a response is streaming to prevent state corruption.

### Per-mode chat history

Fast and GroundedSearch maintain completely independent conversation histories. Switching modes restores that mode's previous conversation. **New Chat** clears only the active mode — the other mode's history is untouched.

### "Analysing results" indicator (GroundedSearch)

Between retrieval attempts, the search header updates to "Analysing results, searching more…" so you can see the model is still working rather than wondering if it's frozen.

### Error recovery

When the API returns an error or times out, the spinner stops, the header shows "Search failed" in red, and a friendly error message appears. Re-asking the same question usually works — free-tier 500 errors are transient.

### New Chat button

Clears only the current mode's conversation and restores the welcome screen with example prompts.

---

## Performance Notes

### Why responses take time

This is an educational project running on **Google AI Studio's free tier**. Responses are intentionally slower than a paid deployment:

```
Wikipedia search + page fetch              ~1–3 s   per search
Gemma generation (GroundedSearch retrieval) ~3–8 s   per attempt (300 token cap)
Gemma generation (answer synthesis)        ~10–20 s (1024 token cap, streamed)
Free-tier API cold start / queue           ~2–10 s  (variable)
```

For a two-search GroundedSearch query: roughly `3 + 8 + 3 + 8 + 15 ≈ 37 seconds` total. Fast mode cuts this to ~10–15s by doing one pass with parallel searches.

### Why Fast mode sometimes says "context does not contain..."

Wikipedia's keyword search is not semantic. If the extracted keywords land on an overview/institution article instead of the specific person-role article, the answer won't be in the retrieved text. The model correctly reports this rather than hallucinating. **Switch to GroundedSearch** — its iterative loop can self-correct with follow-up searches.

### Why the same question sometimes fails then works

Google AI Studio's free tier returns sporadic `500 INTERNAL` errors, especially during peak hours. The engine retries automatically (up to 2 times, 1s/2s backoff). If it still fails, wait 10–15 seconds and re-ask — the error is on Google's side, not in your question.

### Timeout protection

All non-streaming API calls go through `_generate()`, a safe wrapper that:

- Runs the call in a background thread
- Raises a `RuntimeError` after **2 minutes** if no response arrives (prevents infinite spinner)
- **Does not retry timeouts** — a hung call won't un-hang on a second attempt

---

## Free Tier Limits (Google AI Studio)

| Resource                  | Limit     |
| ------------------------- | --------- |
| Requests per minute (RPM) | 15        |
| Tokens per minute (TPM)   | Unlimited |
| Requests per day (RPD)    | 1,500     |

A single GroundedSearch query uses 3–6 API requests (retrieval attempts + answer). Fast mode uses 2 requests (keyword extraction + answer). At 1,500 RPD you can handle ~250–750 queries per day.

---

## Differences from the Original Cookbook

| Aspect         | Cookbook (Claude 2)                     | This project (Gemma 4 31B)                        |
| -------------- | --------------------------------------- | ------------------------------------------------- |
| LLM SDK        | `anthropic` (legacy)                  | `google-genai`                                  |
| Model          | `claude-2`                            | `gemma-4-31b-it`                                |
| Tokenizer      | Anthropic tokenizer for truncation      | Character-based truncation                        |
| Stop sequences | `AI_PROMPT` / `HUMAN_PROMPT` format | `</search_query>` via `GenerateContentConfig` |
| Serving        | Jupyter notebook only                   | Flask app with SSE streaming                      |
| Frontend       | None (stdout)                           | Dark glassmorphism UI with live search progress   |
| Concurrency    | Synchronous                             | Background thread + queue + heartbeat             |
| Search modes   | Single mode                             | Fast (parallel) + GroundedSearch (iterative)      |
| Chat history   | N/A                                     | Per-mode independent conversation history         |
| Error handling | Basic                                   | Retry logic, timeout protection, spinner cleanup  |
| Deployment     | Local only                              | Render-ready (Procfile + gunicorn)                |

---

## Troubleshooting

**`model not found` error**
Check the exact model ID in your AI Studio dashboard and update `GEMMA_MODEL` in `.env`.

**`JSONDecodeError` from Wikipedia**
Wikipedia's API occasionally returns empty responses when rate-limited. The engine retries automatically and logs `[wiki] search() failed`. Wait a few seconds and try again.

**"Network error" in the browser**
Usually means the Flask server crashed. Check the terminal for a Python traceback.

**`ServerError: 500 INTERNAL` / red "Search failed" banner**
Transient free-tier error from Google's API. Re-ask the same question — the retry logic handles single failures automatically; persistent failures clear on the next request.

**"Gemma API timed out (>2 min)"**
The free-tier API occasionally hangs on complex questions with large accumulated context. Re-ask or switch to Fast mode for a quicker answer.

**Answer says "the context does not contain this information"**
Fast mode found a Wikipedia article, but it was the wrong type (e.g. institution overview instead of person-role article). Switch to **🔬 GroundedSearch** which iteratively searches until it finds the right content.

**Answer quality is poor**
Wikipedia's keyword search is not semantic. Try rephrasing with proper nouns, e.g. "Chief Justice of India" instead of "supreme court justice india". GroundedSearch is generally more accurate for complex queries.

---

## Tech Stack

| Layer          | Technology                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------- |
| LLM            | Gemma 4 31B (Google AI Studio, free tier)                                                    |
| LLM SDK        | `google-genai`                                                                             |
| Knowledge base | Wikipedia (`wikipedia` Python library)                                                     |
| Backend        | Flask 3 + Gunicorn (production)                                                              |
| Streaming      | Server-Sent Events (SSE)                                                                     |
| Concurrency    | `threading` (SSE heartbeat) + `concurrent.futures` (parallel wiki fetches + API timeout) |
| Frontend       | Vanilla HTML/CSS/JS, Inter font, Marked.js                                                   |
| Config         | `python-dotenv`                                                                            |
| Deployment     | Render (free tier)                                                                           |
