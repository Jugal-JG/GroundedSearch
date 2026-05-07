"""
GroundedSearch — Flask application entry point.
Run with:  python groundedsearch.py

SSE architecture:
  The RAG engine runs in a background thread and puts events into a queue.
  The Flask route drains that queue and forwards events as SSE.
  While the queue is empty (model is thinking), a heartbeat comment is sent
  every 10 seconds so the browser never sees a silent connection and disconnects.
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

    if not query:
        return {"error": "No query provided"}, 400

    event_queue: queue.Queue = queue.Queue()

    def run_engine():
        """Runs the RAG engine in a background thread, pushes events to the queue."""
        try:
            for event in engine.search(query):
                event_queue.put(event)
        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(None)   # sentinel — tells the SSE loop to stop

    threading.Thread(target=run_engine, daemon=True).start()

    def generate():
        """
        Drains the event queue and yields SSE frames.
        Sends a keep-alive heartbeat comment every 10 s while waiting,
        preventing the browser from closing the idle connection.
        """
        while True:
            try:
                event = event_queue.get(timeout=10)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # SSE comment — browser ignores it but the connection stays alive
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
