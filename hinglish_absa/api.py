"""
REST API for Hinglish Aspect-Based Sentiment Analysis.
Provides endpoints for single and batch inference using Starlette and Uvicorn.
"""

import json
from typing import List
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from hinglish_absa import HinglishABSA

# Global model instance
absa_engine = HinglishABSA()

async def health(request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "hinglish-absa",
        "version": "1.0.0"
    })

async def root(request):
    """API overview and documentation."""
    return JSONResponse({
        "title": "Hinglish Aspect-Based Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check status",
            "/analyze": "POST - Analyze a single sentence {'text': '...'}",
            "/batch": "POST - Analyze multiple sentences {'texts': ['...', '...']}"
        },
        "sample_payload": {
            "text": "Phone ka camera bahut accha hai but battery backup bekar hai."
        }
    })

async def analyze(request):
    """Analyze a single sentence."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "Missing 'text' field in request"}, status_code=400)

    result = absa_engine.analyze(text)
    return JSONResponse(result.to_dict())

async def batch(request):
    """Analyze a list of sentences."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    texts = data.get("texts", [])
    if not isinstance(texts, list):
        return JSONResponse({"error": "'texts' must be a list of strings"}, status_code=400)

    results = absa_engine.analyze_batch(texts)
    return JSONResponse({
        "total": len(results),
        "results": [r.to_dict() for r in results]
    })

routes = [
    Route("/", root, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/analyze", analyze, methods=["POST"]),
    Route("/batch", batch, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    print("Starting Hinglish ABSA REST API on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
