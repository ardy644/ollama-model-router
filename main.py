"""
FastAPI Ollama Model Router
----------------------------
Lightweight orchestration layer for a local Ollama server.
Auto-routes prompts to the best model via keyword heuristics,
enforces VRAM-safe context limits, and strips DeepSeek <think> tags.
"""

import re
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TIMEOUT = 120.0  # seconds

ALLOWED_MODELS = [
    "auto",
    "gemma3:1b",
    "qwen2.5:3b",
    "deepseek-r1:1.5b",
    "qwen2.5-coder:1.5b-base",
]
INSTALLED_MODELS = ALLOWED_MODELS[1:]  # everything except "auto"

# ---------------------------------------------------------------------------
# Regex patterns for auto-routing
# ---------------------------------------------------------------------------

CODING_PATTERN = re.compile(
    r"\b(python|code|function|def|class|sql|bug|script|html|css|javascript|"
    r"typescript|api|endpoint|variable|loop|array|compile|debug|refactor|git|regex)\b",
    re.IGNORECASE,
)

REASONING_PATTERN = re.compile(
    r"\b(solve|calculate|why|step[- ]by[- ]step|logic|evaluate|proof|theorem|"
    r"math|equation|reason|derive|deduce|explain\s+why|analyze)\b",
    re.IGNORECASE,
)

# Pattern to extract <think>...</think> blocks from deepseek-r1 output
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    prompt: str
    model: Literal[
        "auto",
        "gemma3:1b",
        "qwen2.5:3b",
        "deepseek-r1:1.5b",
        "qwen2.5-coder:1.5b-base",
    ] = "auto"
    strip_reasoning: bool = True
    context_limit: int = Field(default=2048, ge=512, le=4096)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def classify_prompt(prompt: str) -> str:
    """Route a prompt to the best model using keyword heuristics.

    Priority: Coding → Reasoning → General fallback (qwen2.5:3b).
    """
    if CODING_PATTERN.search(prompt):
        return "qwen2.5-coder:1.5b-base"
    if REASONING_PATTERN.search(prompt):
        return "deepseek-r1:1.5b"
    return "qwen2.5:3b"


def process_reasoning(
    response_text: str, strip: bool
) -> tuple[str, Optional[str]]:
    """Extract and optionally strip <think> tags from deepseek-r1 output.

    Returns (cleaned_text, reasoning_trace | None).
    """
    match = THINK_TAG_PATTERN.search(response_text)
    reasoning = match.group(1).strip() if match else None

    if strip and match:
        cleaned = THINK_TAG_PATTERN.sub("", response_text).strip()
        return cleaned, reasoning
    return response_text, reasoning


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ollama Model Router",
    description="Lightweight local router for Ollama with auto-routing, "
    "VRAM-safe context limits, and DeepSeek think-tag processing.",
    version="1.0.0",
)


# ---- POST /v1/chat -------------------------------------------------------


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    """Send a prompt to Ollama, optionally auto-routing to the best model."""

    # 1. Resolve target model
    if request.model == "auto":
        target_model = classify_prompt(request.prompt)
    else:
        if request.model not in INSTALLED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. "
                f"Available: {INSTALLED_MODELS}",
            )
        target_model = request.model

    # 2. Build Ollama payload (always enforce num_ctx for VRAM safety)
    payload = {
        "model": target_model,
        "prompt": request.prompt,
        "stream": False,
        "options": {"num_ctx": request.context_limit},
    }

    # 3. Forward to Ollama
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(OLLAMA_GENERATE, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama server is offline or unreachable at "
            f"{OLLAMA_BASE}. Please start Ollama and try again.",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {exc.response.status_code} — "
            f"{exc.response.text}",
        )

    data = resp.json()
    response_text: str = data.get("response", "")

    # 4. Process <think> tags for deepseek-r1
    reasoning = None
    if target_model == "deepseek-r1:1.5b":
        response_text, reasoning = process_reasoning(
            response_text, request.strip_reasoning
        )

    return {
        "model": target_model,
        "response": response_text,
        "reasoning": reasoning,
        "tokens": {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        },
    }


# ---- GET /v1/models ------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    """Return the list of installed models and auto-routing rules."""
    return {
        "installed_models": INSTALLED_MODELS,
        "routing_options": ALLOWED_MODELS,
        "auto_routing_rules": {
            "coding_keywords": (
                "python, code, function, def, class, sql, bug, script, "
                "html, css, javascript, typescript, api, endpoint, variable, "
                "loop, array, compile, debug, refactor, git, regex"
            ),
            "reasoning_keywords": (
                "solve, calculate, why, step-by-step, logic, evaluate, "
                "proof, theorem, math, equation, reason, derive, deduce, "
                "explain why, analyze"
            ),
            "default_fallback": "qwen2.5:3b",
        },
    }


# ---- GET /health ----------------------------------------------------------


@app.get("/health")
async def health():
    """Check service health and Ollama connectivity."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass
    return {"status": "healthy", "ollama_connected": ollama_ok}
