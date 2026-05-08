"""
GroundedSearch — Flask application entry point.
Run with:  python groundedsearch.py

Two modes:
  grounded  — iterative Wikipedia retrieval loop (thorough, 25-45s)
  fast      — single Wikipedia search + direct answer  (~10-15s)
"""

import json
import queue
import threading
from flask import Flask, render_template, request, Response, stream_with_context
from rag_engine import GroundedSearchEngine

app = Flask(__name__)
engine = GroundedSearchEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    mode  = data.get("mode", "grounded")   # "grounded" | "fast"

    if not query:
        return {"error": "No query provided"}, 400

    event_queue: queue.Queue = queue.Queue()

    def run_engine():
        generator = engine.fast_search(query) if mode == "fast" else engine.search(query)
        try:
            for event in generator:
                event_queue.put(event)
        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=run_engine, daemon=True).start()

    def generate():
        while True:
            try:
                event = event_queue.get(timeout=10)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
