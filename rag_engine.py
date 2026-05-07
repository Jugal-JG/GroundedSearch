"""
GroundedSearch RAG Engine
Adapted from the Wikipedia Search Cookbook (originally built for Claude 2).

Architecture preserved from cookbook:
  1. Tool-description prompt  — tells the model how to call the search tool
  2. Retrieval loop           — model emits <search_query> tags; we intercept,
                                run Wikipedia, inject <search_results>, repeat
  3. Answer synthesis         — extracted <information> is fed to a clean prompt
                                so the model focuses entirely on composing the answer

Changes from cookbook:
  - Claude 2 (Anthropic SDK)  →  Gemma 4 31B (google-genai SDK)
  - Anthropic tokenizer removed; character-based page truncation used instead
  - Class restructured as a generator so Flask can stream SSE events to the browser
"""

import re
import os
import time
import wikipedia
from google import genai
from google.genai import types
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Set a proper user-agent so Wikipedia API doesn't rate-limit us
wikipedia.set_user_agent("GroundedSearch/1.0 (educational RAG project)")

# ---------------------------------------------------------------------------
# Prompts  — kept verbatim from the cookbook, with stronger atomic-query rule
# ---------------------------------------------------------------------------

# Cookbook cell 3 — tool description
# Key addition: concrete BAD vs GOOD examples to stop Gemma from generating
# compound sentence queries like "release date of movie A and movie B"
TOOL_DESCRIPTION = """You will be asked a question by a human user. You have access to the following tool to help answer the question.
<tool_description>
Search Engine Tool
* The search engine searches Wikipedia. It returns each page's title and content.
* Queries MUST be short keywords — 1 to 4 words only. Never write a full sentence.
* If the question involves multiple things, search for ONE thing at a time.

CORRECT examples:
  <search_query>Oppenheimer film</search_query>
  <search_query>neural network</search_query>
  <search_query>Eiffel Tower</search_query>

WRONG examples (never do this):
  <search_query>what is the release date of Oppenheimer film</search_query>
  <search_query>Oppenheimer and Barbie release dates</search_query>

* Use the syntax: <search_query>keywords</search_query>
* Results come back in <search_results> tags.
</tool_description>"""

# Cookbook cell 6 — retrieval prompt
RETRIEVAL_PROMPT = """Before researching, think briefly inside <scratchpad> tags (2-3 sentences max).
IMPORTANT: If the question asks about multiple things, search for each ONE AT A TIME.

After each search, write ONE sentence inside <search_quality></search_quality> tags — just say if you need more info.
When you have everything, write the key facts inside <information></information> tags (bullet points, be concise).
Otherwise issue another search immediately.

Question: <question>{query}</question>

Remember: <search_query> must contain only 1-4 keywords. Keep all responses SHORT."""

# Cookbook cell 8 — answer synthesis prompt
ANSWER_PROMPT = """Here is a user query: <query>{query}</query>.
Here is some relevant information retrieved from Wikipedia: <information>{information}</information>.
Please answer the question using the relevant information. Be clear, accurate, and well-structured. Where helpful, use bullet points or short paragraphs."""


# ---------------------------------------------------------------------------
# Wikipedia search tool  — adapted from cookbook's WikipediaSearchTool
# ---------------------------------------------------------------------------

@dataclass
class WikipediaSource:
    title: str
    content: str
    url: str


class WikipediaSearchTool:
    """
    Adapted from cookbook's WikipediaSearchTool.
    20,000 chars is fine for Gemma 4 31B (256k context window, fast prefill).
    20,000 chars is fine — Gemma 4 31B prefills fast. The real bottleneck is generation
    length: if the model writes a verbose <search_quality> section (400-800 tokens) before
    the next <search_query>, that is 10-53 seconds of silence at 15-40 tps.
    """

    def __init__(self, truncate_chars: int = 20_000):
        self.truncate_chars = truncate_chars

    def search(self, query: str, n_results: int = 1) -> tuple[list[WikipediaSource], str]:
        """Run a Wikipedia search and return sources + formatted XML string."""

        # Small delay between calls to avoid Wikipedia rate-limiting
        time.sleep(0.3)

        try:
            raw_titles: list[str] = wikipedia.search(query)
        except Exception as exc:
            print(f"[wiki] search() failed for '{query}': {type(exc).__name__}: {exc}")
            return [], self._format([])

        sources: list[WikipediaSource] = []
        for title in raw_titles:
            if len(sources) >= n_results:
                break
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except Exception:
                continue

            sources.append(WikipediaSource(
                title=page.title,
                content=page.content[: self.truncate_chars],
                url=page.url,
            ))

        print(f"[wiki] '{query}' -> {[s.title for s in sources]}")
        return sources, self._format(sources)

    def _format(self, sources: list[WikipediaSource]) -> str:
        """Matches the cookbook's wrap_search_results / search_results_to_string format."""
        if not sources:
            return "\n<search_results>\nNo results found. Try a different, shorter search term.\n</search_results>"
        items = "\n".join(
            f'<item index="{i + 1}">\n<page_content>\n'
            f'Page Title: {s.title}\nPage Content:\n{s.content}\n'
            f'</page_content>\n</item>'
            for i, s in enumerate(sources)
        )
        return f"\n<search_results>\n{items}\n</search_results>"


# ---------------------------------------------------------------------------
# Grounded search engine  — replaces cookbook's ClientWithRetrieval
# ---------------------------------------------------------------------------

class GroundedSearchEngine:
    """
    Drives the three-phase pipeline from the cookbook:
      phase 1 — retrieval loop (search_query intercept)
      phase 2 — answer synthesis from extracted <information>
    """

    def __init__(self, max_searches: int = 5):
        api_key = os.environ.get("GEMINI_API_KEY")
        self.model_id = os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")
        self.client = genai.Client(api_key=api_key)
        self.search_tool = WikipediaSearchTool()
        self.max_searches = max_searches

    def search(self, query: str):
        """
        Generator — yields SSE event dicts consumed by Flask's /chat route.

        Event types:
          search_start  {"query": str}
          search_done   {"title": str, "url": str}
          search_empty  {"query": str}            ← new: fired when Wikipedia finds nothing
          answer_start  {}
          token         {"content": str}
          sources       {"sources": [...]}
          done          {}
          error         {"message": str}
        """
        try:
            all_sources, information = yield from self._retrieval_phase(query)
            yield from self._answer_phase(query, information, all_sources)
        except Exception as exc:
            print(f"[engine] unhandled: {type(exc).__name__}: {exc}")
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Phase 1 — retrieval loop (cookbook's retrieve() method)
    # ------------------------------------------------------------------

    def _retrieval_phase(self, query: str):
        prompt = f"{TOOL_DESCRIPTION}\n\n{RETRIEVAL_PROMPT.format(query=query)}\n\n"
        accumulated = ""
        all_sources: list[WikipediaSource] = []

        for attempt in range(self.max_searches):
            print(f"[retrieval] attempt {attempt + 1}")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt + accumulated,
                config=types.GenerateContentConfig(
                    stop_sequences=["</search_query>"],
                    # 300 tokens caps generation at 7-20s (at 15-40 tps).
                    # Enough for scratchpad+search_query (~120 tokens) or a
                    # brief <information> summary. Prevents the model from writing
                    # 600-token verbose <search_quality> sections that cause 40s+ waits.
                    max_output_tokens=300,
                    temperature=0.1,
                ),
            )

            chunk = response.text or ""
            finish = response.candidates[0].finish_reason
            print(f"[retrieval] finish={finish}, got: {chunk[:120]!r}")
            accumulated += chunk

            search_requested = (
                "<search_query>" in chunk and "</search_query>" not in chunk
            )

            if search_requested:
                idx = chunk.rfind("<search_query>") + len("<search_query>")
                search_term = chunk[idx:].strip()

                if not search_term:
                    break

                print(f"[retrieval] search_term: {search_term!r}")
                yield {"type": "search_start", "query": search_term}

                sources, result_xml = self.search_tool.search(search_term, n_results=1)
                all_sources.extend(sources)
                accumulated += "</search_query>" + result_xml

                if sources:
                    for src in sources:
                        yield {"type": "search_done", "title": src.title, "url": src.url}
                else:
                    # Bug fix: always resolve the spinner — never leave it hanging
                    yield {"type": "search_empty", "query": search_term}
            else:
                break

        info_matches = re.findall(
            r"<information>(.*?)</information>", accumulated, re.DOTALL
        )
        information = info_matches[-1].strip() if info_matches else accumulated
        print(f"[retrieval] done. sources={len(all_sources)}, info_len={len(information)}")

        return all_sources, information

    # ------------------------------------------------------------------
    # Phase 2 — answer synthesis
    # ------------------------------------------------------------------

    def _answer_phase(self, query: str, information: str, all_sources: list[WikipediaSource]):
        answer_prompt = ANSWER_PROMPT.format(query=query, information=information)

        yield {"type": "answer_start"}
        print("[answer] streaming...")

        for chunk in self.client.models.generate_content_stream(
            model=self.model_id,
            contents=answer_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.2,
            ),
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield {"type": "token", "content": text}

        if all_sources:
            yield {
                "type": "sources",
                "sources": [{"title": s.title, "url": s.url} for s in all_sources],
            }

        yield {"type": "done"}
        print("[answer] done.")
