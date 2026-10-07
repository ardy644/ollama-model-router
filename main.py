"""
FastAPI Ollama Model Router
----------------------------
Lightweight orchestration layer for a local Ollama server.
Auto-routes prompts to the best model via keyword heuristics,
enforces VRAM-safe context limits, and strips DeepSeek <think> tags.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT = f"{OLLAMA_BASE}/api/chat"
OLLAMA_TIMEOUT = 120.0  # seconds

DEFAULT_MODEL_CHOICES = [
    "gemma3:1b",
    "qwen2.5:3b",
    "deepseek-r1:1.5b",
    "qwen2.5-coder:1.5b-base",
]
SETTINGS_PATH = Path(__file__).with_name("router_settings.json")

# ---------------------------------------------------------------------------
# Persona & System Prompt Presets
# ---------------------------------------------------------------------------

PERSONAS = {
    "general": {
        "name": "General Assistant",
        "description": "Balanced, polite, and helpful general-purpose AI.",
        "system": "You are a helpful, respectful, and honest AI assistant.",
        "temperature": 0.7,
    },
    "coder": {
        "name": "Senior Software Engineer",
        "description": "Production-grade code, strict type hints, zero filler.",
        "system": (
            "You are a principal software engineer. Provide clean, secure, production-grade code with "
            "precise type hints and docstrings. Do not include conversational filler or pleasantries."
        ),
        "temperature": 0.2,
    },
    "reasoner": {
        "name": "Math & Logic Specialist",
        "description": "Rigorous step-by-step mathematical reasoning.",
        "system": (
            "You are a mathematical and logical reasoning specialist. Work methodically step by step, "
            "verify every intermediate step, and clearly explain proofs and derivations."
        ),
        "temperature": 0.3,
    },
    "executive": {
        "name": "Executive Summarizer",
        "description": "High-density bullet points, bold metrics, zero fluff.",
        "system": (
            "You are an executive chief of staff. Deliver answers with extreme brevity. "
            "Use bullet points, bold key takeaways, and eliminate all conversational fluff."
        ),
        "temperature": 0.3,
    },
}

# Pattern to extract <think>...</think> blocks from deepseek-r1 output
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    prompt: str
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)
    model: str = "auto"
    persona: Optional[Literal["general", "coder", "reasoner", "executive", "custom"]] = "general"
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    stream: bool = False
    strip_reasoning: bool = True
    context_limit: int = Field(default=2048, ge=512, le=4096)
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = "auto"
    messages: list[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    stream: Optional[bool] = False
    context_limit: Optional[int] = Field(default=2048, ge=512, le=4096)
    persona: Optional[str] = "general"


class RouterSettings(BaseModel):
    model_choices: list[str] = Field(default_factory=lambda: list(DEFAULT_MODEL_CHOICES), min_length=1, max_length=40)
    coding_model: str = "qwen2.5-coder:1.5b-base"
    reasoning_model: str = "deepseek-r1:1.5b"
    fallback_model: str = "gemma3:1b"
    coding_keywords: list[str] = Field(default_factory=lambda: [
        "python", "code", "function", "def", "class", "sql", "bug", "script", "html", "css",
        "javascript", "typescript", "api", "endpoint", "variable", "loop", "array", "compile",
        "debug", "refactor", "git", "regex",
    ], min_length=1, max_length=100)
    reasoning_keywords: list[str] = Field(default_factory=lambda: [
        "solve", "calculate", "why", "step-by-step", "step by step", "logic", "evaluate", "proof",
        "theorem", "math", "equation", "reason", "derive", "deduce", "explain why", "analyze",
    ], min_length=1, max_length=100)

    @field_validator("model_choices")
    @classmethod
    def validate_model_choices(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not cleaned or "auto" in cleaned or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value)
            for value in cleaned
        ):
            raise ValueError("Model choices must be valid Ollama model tags")
        return cleaned

    @field_validator("coding_keywords", "reasoning_keywords")
    @classmethod
    def validate_keywords(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value.strip().lower() for value in values if value.strip()))
        if not cleaned or any(len(value) > 60 for value in cleaned):
            raise ValueError("Keyword lists must contain non-empty terms of at most 60 characters")
        return cleaned

    @field_validator("coding_model", "reasoning_model", "fallback_model")
    @classmethod
    def validate_routing_model_tags(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", value):
            raise ValueError("Routing model must be a valid Ollama model tag")
        return value

    @model_validator(mode="after")
    def validate_routing_models_are_choices(self):
        missing = {self.coding_model, self.reasoning_model, self.fallback_model} - set(self.model_choices)
        if missing:
            raise ValueError(f"Routing models must be included in model_choices: {sorted(missing)}")
        return self


def load_router_settings() -> RouterSettings:
    try:
        if SETTINGS_PATH.exists():
            return RouterSettings.model_validate_json(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger(__name__).exception("Could not load router settings; using defaults")
    return RouterSettings()


def save_router_settings(settings: RouterSettings) -> None:
    temporary_path = SETTINGS_PATH.with_suffix(".tmp")
    temporary_path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(SETTINGS_PATH)


ROUTER_SETTINGS = load_router_settings()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def classify_prompt(prompt: str) -> str:
    """Route with current settings. Coding rules take priority over reasoning."""
    coding_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in ROUTER_SETTINGS.coding_keywords) + r")(?!\w)",
        re.IGNORECASE,
    )
    reasoning_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in ROUTER_SETTINGS.reasoning_keywords) + r")(?!\w)",
        re.IGNORECASE,
    )
    if coding_pattern.search(prompt):
        return ROUTER_SETTINGS.coding_model
    if reasoning_pattern.search(prompt):
        return ROUTER_SETTINGS.reasoning_model
    return ROUTER_SETTINGS.fallback_model


def format_prompt_with_history(
    prompt: str, history: list[ChatHistoryMessage], max_history_chars: int = 6000
) -> str:
    """Add recent chat turns while keeping the extra prompt within a small bound."""
    recent = []
    remaining = max_history_chars
    for message in reversed(history):
        content = message.content.strip()
        if not content or remaining <= 0:
            continue
        content = content[-remaining:]
        recent.append((message.role, content))
        remaining -= len(content)
    recent.reverse()
    if not recent:
        return prompt
    transcript = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {content}"
        for role, content in recent
    )
    return f"Previous conversation:\n{transcript}\nUser: {prompt}"


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
# FastAPI Application & Lifespan Connection Pooling
# ---------------------------------------------------------------------------


async def ollama_is_reachable(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get(f"{OLLAMA_BASE}/", timeout=1.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def start_ollama_if_needed(app: FastAPI, client: httpx.AsyncClient) -> None:
    if await ollama_is_reachable(client):
        return
    executable = shutil.which("ollama")
    if not executable and os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.is_file():
                executable = str(candidate)
    if not executable:
        logging.getLogger(__name__).warning(
            "Ollama is not running and its CLI was not found; start Ollama manually."
        )
        return
    try:
        process_options: dict[str, object] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            process_options["start_new_session"] = True
        app.state.ollama_process = subprocess.Popen(
            [executable, "serve"], **process_options
        )
        logging.getLogger(__name__).info("Started Ollama server automatically.")
        for _ in range(20):
            if await ollama_is_reachable(client):
                return
            await asyncio.sleep(0.5)
        logging.getLogger(__name__).warning(
            "Ollama was launched but did not become available within 10 seconds."
        )
    except OSError:
        logging.getLogger(__name__).exception("Could not start Ollama automatically")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lifecycle of shared HTTP client with connection pooling."""
    limits = httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(
        timeout=OLLAMA_TIMEOUT,
        connect=10.0,
    )
    client = httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers={"User-Agent": "Ollama-Router/1.0"},
    )
    app.state.http_client = client
    try:
        await start_ollama_if_needed(app, client)
        yield
    finally:
        await client.aclose()


async def get_http_client(request: Request) -> AsyncIterator[httpx.AsyncClient]:
    """Dependency provider for shared HTTPX async client."""
    client: Optional[httpx.AsyncClient] = getattr(request.app.state, "http_client", None)
    if client is not None and not client.is_closed:
        yield client
        return
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as fallback_client:
        yield fallback_client


app = FastAPI(
    title="Ollama Model Router",
    description="Lightweight local router for Ollama with auto-routing, "
    "VRAM-safe context limits, and DeepSeek think-tag processing.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log unexpected errors while returning a generic response to clients."""
    logging.getLogger(__name__).error(
        "Unhandled error while processing %s", request.url.path, exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred.",
            "path": request.url.path,
        },
    )


# ---------------------------------------------------------------------------
# Interactive Web UI
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ollama Model Router</title>
  <!-- Marked.js for Markdown & Highlight.js for Code Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
  <style>
    :root {
      --bg-main: #212121;
      --bg-sidebar: #171717;
      --bg-card: #2f2f2f;
      --bg-input: #2f2f2f;
      --border-color: #383838;
      --text-main: #ececec;
      --text-muted: #b4b4b4;
      --accent: #10a37f;
      --accent-hover: #1a7f64;
      --gemini-gradient: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }

    /* Sidebar */
    #sidebar {
      width: 260px;
      background-color: var(--bg-sidebar);
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--border-color);
      transition: all 0.25s ease;
      z-index: 10;
    }
    #sidebar.collapsed { width: 0; min-width: 0; overflow: hidden; border: none; }
    
    .sidebar-header { padding: 14px 16px; display: flex; align-items: center; justify-content: space-between; }
    .new-chat-btn {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      padding: 10px 14px;
      background: transparent;
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }
    .new-chat-btn:hover { background: #262626; }

    .chat-history { flex: 1; overflow-y: auto; padding: 10px 12px; display: flex; flex-direction: column; gap: 4px; }
    .history-title { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); padding: 8px 8px 4px; letter-spacing: 0.5px; font-weight: 600; }
    .history-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 9px 12px;
      border-radius: 8px;
      font-size: 0.88rem;
      color: var(--text-main);
      cursor: pointer;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      transition: background 0.15s;
    }
    .history-item:hover, .history-item.active { background: #262626; }
    .history-item span { overflow: hidden; text-overflow: ellipsis; }
    .del-chat { color: #888; font-size: 0.8rem; display: none; margin-left: 8px; }
    .history-item:hover .del-chat { display: inline; }
    .del-chat:hover { color: #f85149; }

    .sidebar-footer { padding: 14px 16px; border-top: 1px solid var(--border-color); font-size: 0.8rem; display: flex; flex-direction: column; gap: 8px; }
    .hardware-pill { display: flex; align-items: center; gap: 8px; color: var(--text-muted); }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #666; }
    .status-dot.online { background: #10a37f; box-shadow: 0 0 8px rgba(16,163,127,0.6); }
    .status-dot.offline { background: #f85149; }

    /* Main Container */
    #main { flex: 1; display: flex; flex-direction: column; height: 100vh; position: relative; }

    /* Top Nav */
    .top-nav {
      height: 54px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      border-bottom: 1px solid transparent;
      z-index: 5;
    }
    .nav-left { display: flex; align-items: center; gap: 12px; }
    .icon-btn { background: transparent; border: none; color: var(--text-main); cursor: pointer; padding: 8px; border-radius: 8px; display: flex; align-items: center; }
    .icon-btn:hover { background: #2f2f2f; }

    .model-selector {
      background: #2a2a2a;
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 12px;
      border-radius: 12px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      outline: none;
    }
    .model-selector:hover { background: #333; }
    .model-selector option { background: #212121; color: #fff; font-weight: normal; }

    .nav-right { display: flex; align-items: center; gap: 10px; }
    .pill-link { color: var(--text-muted); text-decoration: none; font-size: 0.85rem; padding: 6px 10px; border-radius: 8px; }
    .pill-link:hover { color: var(--text-main); background: #2a2a2a; }

    /* Chat Area */
    #chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px 140px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .chat-inner { width: 100%; max-width: 780px; display: flex; flex-direction: column; gap: 24px; }

    /* Welcome Hero (ChatGPT / Gemini style) */
    .hero { margin: 60px auto 30px; text-align: center; max-width: 600px; display: flex; flex-direction: column; align-items: center; }
    .hero-title {
      font-size: 2.2rem;
      font-weight: 700;
      background: var(--gemini-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    .hero-sub { color: var(--text-muted); font-size: 1.05rem; margin-bottom: 32px; }
    .hero-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%; }
    .hero-card {
      background: #2a2a2a;
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 16px;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s;
    }
    .hero-card:hover { background: #333; border-color: #555; transform: translateY(-2px); }
    .hero-card-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; color: #fff; }
    .hero-card-desc { font-size: 0.8rem; color: var(--text-muted); }

    /* Messages */
    .msg-row { display: flex; width: 100%; gap: 16px; line-height: 1.6; }
    .msg-row.user { justify-content: flex-end; }
    
    .msg-user-bubble {
      background: #2f2f2f;
      padding: 12px 18px;
      border-radius: 20px;
      max-width: 80%;
      color: #fff;
      font-size: 0.95rem;
      white-space: pre-wrap;
    }

    .msg-bot-content {
      flex: 1;
      max-width: 100%;
      overflow-x: auto;
      font-size: 0.96rem;
    }
    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: var(--gemini-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      flex-shrink: 0;
      color: #fff;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }

    .bot-header-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #282828;
      border: 1px solid var(--border-color);
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.75rem;
      color: #79c0ff;
      margin-bottom: 12px;
      font-family: monospace;
    }

    /* Thinking Drawer */
    details.think-box {
      background: #1c1c1c;
      border: 1px solid #333;
      border-radius: 10px;
      padding: 10px 14px;
      margin-bottom: 16px;
      font-size: 0.85rem;
      color: #999;
    }
    details.think-box summary {
      cursor: pointer;
      color: #d2a8ff;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    details.think-box pre {
      margin-top: 10px;
      white-space: pre-wrap;
      font-family: inherit;
      color: #b0b0b0;
      border-left: 2px solid #58a6ff;
      padding-left: 10px;
    }

    /* Markdown & Code Blocks */
    .msg-bot-content h1, .msg-bot-content h2, .msg-bot-content h3 { margin: 16px 0 8px; color: #fff; }
    .msg-bot-content p { margin-bottom: 12px; }
    .msg-bot-content ul, .msg-bot-content ol { margin: 0 0 12px 20px; }
    .msg-bot-content li { margin-bottom: 4px; }
    .msg-bot-content pre {
      background: #161616;
      border: 1px solid #333;
      border-radius: 10px;
      overflow: hidden;
      margin: 14px 0;
    }
    .code-header {
      background: #252525;
      padding: 6px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: #aaa;
      font-family: monospace;
    }
    .copy-btn {
      background: transparent;
      border: none;
      color: #aaa;
      cursor: pointer;
      font-size: 0.75rem;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .copy-btn:hover { color: #fff; }
    .msg-bot-content pre code {
      display: block;
      padding: 14px;
      overflow-x: auto;
      font-family: "Fira Code", Consolas, Monaco, monospace;
      font-size: 0.88rem;
    }
    p code {
      background: #2e2e2e;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.88rem;
    }

    /* Floating Input */
    .input-wrapper {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      padding: 0 16px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      background: linear-gradient(180deg, transparent 0%, var(--bg-main) 30%);
      pointer-events: none;
    }
    .input-box {
      width: 100%;
      max-width: 780px;
      background: var(--bg-input);
      border: 1px solid #444;
      border-radius: 26px;
      padding: 10px 16px;
      display: flex;
      align-items: flex-end;
      gap: 10px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      pointer-events: auto;
      transition: border-color 0.15s;
    }
    .input-box:focus-within { border-color: #666; }
    textarea#prompt-input {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-main);
      font-size: 0.95rem;
      outline: none;
      resize: none;
      max-height: 200px;
      height: 28px;
      line-height: 24px;
      padding: 2px 4px;
    }
    button.send-btn {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      background: #444;
      border: none;
      color: #999;
      cursor: not-allowed;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      transition: all 0.15s;
      flex-shrink: 0;
    }
    button.send-btn.active {
      background: #fff;
      color: #000;
      cursor: pointer;
    }
    button.send-btn.active:hover { background: #e0e0e0; }
    .stop-btn {
      display: none;
      align-items: center;
      gap: 6px;
      background: #2a2a2a;
      border: 1px solid #444;
      color: #ececec;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 6px 14px;
      border-radius: 18px;
      margin-bottom: 8px;
      cursor: pointer;
      pointer-events: auto;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      transition: all 0.15s ease;
    }
    .stop-btn:hover {
      background: #383838;
      border-color: #666;
      color: #ff7b72;
    }
    .stop-btn svg {
      color: #ff7b72;
    }
    .input-footer {
      font-size: 0.72rem;
      color: #777;
      margin-top: 8px;
      text-align: center;
      pointer-events: auto;
    }

    /* Modal for Settings */
    .modal-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: none;
      align-items: center; justify-content: center; z-index: 100;
    }
    .modal-overlay.open { display: flex; }
    .modal {
      background: #252525;
      border: 1px solid var(--border-color);
      border-radius: 16px;
      width: 90%;
      max-width: 560px;
      max-height: 85vh;
      overflow-y: auto;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modal-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    .modal-close { background: transparent; border: none; color: #aaa; cursor: pointer; font-size: 18px; }
    .setting-row { margin-bottom: 16px; }
    .setting-label { font-size: 0.88rem; font-weight: 500; margin-bottom: 6px; display: flex; justify-content: space-between; }
    .setting-desc { font-size: 0.75rem; color: #999; margin-top: 4px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .advanced-panel { display: none; border-top: 1px solid var(--border-color); padding-top: 16px; margin-top: 4px; }
    .advanced-panel.open { display: block; }
    .setting-textarea {
      width: 100%; min-height: 110px; resize: vertical; box-sizing: border-box;
      background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 8px;
      color: var(--text-main); padding: 10px; font: inherit; font-size: 0.84rem;
    }
    .setting-input, .setting-select {
      width: 100%; box-sizing: border-box; background: var(--bg-input);
      border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-main);
      padding: 9px 10px; font: inherit; font-size: 0.84rem;
    }
    .save-settings-btn {
      border: 0; border-radius: 8px; padding: 9px 14px; color: white;
      background: var(--accent); cursor: pointer; font-weight: 600;
    }
  </style>
</head>
<body>
  <!-- Sidebar -->
  <aside id="sidebar">
    <div class="sidebar-header">
      <button class="new-chat-btn" onclick="startNewChat()">
        <span>＋</span> New Chat
      </button>
    </div>
    <div class="chat-history">
      <div class="history-title">Recent Chats</div>
      <div id="history-list"></div>
    </div>
    <div class="sidebar-footer">
      <div class="hardware-pill">
        <span class="status-dot" id="sidebar-dot"></span>
        <span id="sidebar-status">Checking Ollama...</span>
      </div>
      <div style="color: #777; font-size: 0.72rem;">NVIDIA GTX 1650 • 4 GB VRAM Budget</div>
    </div>
  </aside>

  <!-- Main Chat Section -->
  <main id="main">
    <nav class="top-nav">
      <div class="nav-left">
        <button class="icon-btn" onclick="toggleSidebar()" title="Toggle Sidebar">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
        </button>
        <select class="model-selector" id="model-select">
          <option value="auto">✨ Auto Router (Smart Classifier)</option>
          <option value="qwen2.5-coder:1.5b-base">💻 Qwen 2.5 Coder 1.5B (Code)</option>
          <option value="deepseek-r1:1.5b">🧠 DeepSeek R1 1.5B (Reasoning)</option>
          <option value="qwen2.5:3b">💬 Qwen 2.5 3B (General Writing)</option>
          <option value="gemma3:1b">⚡ Gemma 3 1B (Ultra-fast Fallback)</option>
        </select>
        <select class="model-selector" id="persona-select" title="Model Customization Persona">
          <option value="general">🎭 Persona: General</option>
          <option value="coder">💻 Persona: Senior Coder</option>
          <option value="reasoner">🧠 Persona: Math & Logic</option>
          <option value="executive">⚡ Persona: Executive Brief</option>
        </select>
      </div>
      <div class="nav-right">
        <button class="icon-btn" onclick="openSettings()" title="Settings">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        </button>
        <a href="/docs" target="_blank" class="pill-link">API Docs ↗</a>
      </div>
    </nav>

    <!-- Chat Messages -->
    <div id="chat-scroll">
      <div class="chat-inner" id="chat-inner">
        <!-- Hero Section -->
        <div class="hero" id="hero-section">
          <div class="hero-title">Hello, Aryan</div>
          <div class="hero-sub">What would you like to explore today?</div>
          <div class="hero-grid">
            <div class="hero-card" onclick="fillAndSend('Write a Python function to reverse a string and write unit tests')">
              <div class="hero-card-title">💻 Code Generation</div>
              <div class="hero-card-desc">Auto-routes to Qwen 2.5 Coder</div>
            </div>
            <div class="hero-card" onclick="fillAndSend('Solve step-by-step: If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take?')">
              <div class="hero-card-title">🧠 Math & Logic</div>
              <div class="hero-card-desc">Auto-routes to DeepSeek R1 with reasoning</div>
            </div>
            <div class="hero-card" onclick="fillAndSend('Summarize the advantages of running local models over cloud APIs')">
              <div class="hero-card-title">💬 Deep Analysis</div>
              <div class="hero-card-desc">Auto-routes to Qwen 2.5 3B</div>
            </div>
            <div class="hero-card" onclick="fillAndSend('Say hello in French, German, and Japanese')">
              <div class="hero-card-title">⚡ Quick Translation</div>
              <div class="hero-card-desc">Fast response fallback</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Input Box -->
    <div class="input-wrapper">
      <button class="stop-btn" id="stop-btn" onclick="stopGenerating()" title="Cancel generation">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
        <span>Stop Generating</span>
      </button>
      <div class="input-box">
        <textarea id="prompt-input" rows="1" placeholder="Message Ollama Model Router..."></textarea>
        <button class="send-btn" id="send-btn" onclick="submitMessage()">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
        </button>
      </div>
      <div class="input-footer">
        Ollama Model Router • Enforces 2048 num_ctx to prevent GTX 1650 VRAM overflow
      </div>
    </div>
  </main>

  <!-- Settings Modal -->
  <div class="modal-overlay" id="settings-modal" onclick="closeSettings(event)">
    <div class="modal" onclick="event.stopPropagation()">
      <div class="modal-title">
        <span>Router Settings</span>
        <button class="modal-close" onclick="closeSettings()">&times;</button>
      </div>
      <div class="setting-row">
        <div class="setting-label">Routing rules and models</div>
        <label class="setting-desc" for="model-choices-input">Model choices (comma separated)</label>
        <input class="setting-input" id="model-choices-input" type="text" maxlength="3000" placeholder="gemma3:1b, qwen2.5:3b">
        <div class="setting-desc">Use model tags available in Ollama. Each routing model must be in this list.</div>
      </div>
      <div class="setting-row">
        <label class="setting-label" for="coding-model-select">Model for coding prompts</label>
        <select class="setting-select" id="coding-model-select"></select>
      </div>
      <div class="setting-row">
        <label class="setting-label" for="reasoning-model-select">Model for reasoning prompts</label>
        <select class="setting-select" id="reasoning-model-select"></select>
      </div>
      <div class="setting-row">
        <label class="setting-label" for="fallback-model-select">Model for other prompts</label>
        <select class="setting-select" id="fallback-model-select"></select>
      </div>
      <div class="setting-row">
        <label class="setting-label" for="coding-keywords-input">Coding keywords (comma separated)</label>
        <textarea class="setting-textarea" id="coding-keywords-input" maxlength="6000" rows="3"></textarea>
      </div>
      <div class="setting-row">
        <label class="setting-label" for="reasoning-keywords-input">Reasoning keywords (comma separated)</label>
        <textarea class="setting-textarea" id="reasoning-keywords-input" maxlength="6000" rows="3"></textarea>
      </div>
      <div class="setting-row">
        <button class="save-settings-btn" id="save-routing-settings">Save routing settings</button>
        <span class="setting-desc" id="routing-settings-status" role="status" aria-live="polite"></span>
      </div>
      <div class="setting-row">
        <div class="setting-label">
          <span>Context Window (num_ctx)</span>
          <span id="modal-ctx-val" style="font-family: monospace; color: #79c0ff;">2048</span>
        </div>
        <input type="range" id="modal-ctx-slider" min="512" max="4096" step="256" value="2048">
        <div class="setting-desc">Enforces KV cache limit to prevent memory spilling into system RAM.</div>
      </div>
      <div class="setting-row" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <div class="setting-label" style="margin-bottom: 2px;">Extract Reasoning Tags</div>
          <div class="setting-desc">Strips &lt;think&gt; tags from DeepSeek into a collapsible drawer.</div>
        </div>
        <input type="checkbox" id="modal-strip-chk" checked style="accent-color: var(--accent); width: 18px; height: 18px;">
      </div>
      <div class="setting-row" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <div class="setting-label" style="margin-bottom: 2px;">Advanced customization</div>
          <div class="setting-desc">Set custom instructions and response creativity.</div>
        </div>
        <input type="checkbox" id="advanced-mode-chk" style="accent-color: var(--accent); width: 18px; height: 18px;">
      </div>
      <div class="advanced-panel" id="advanced-panel">
        <div class="setting-row">
          <div class="setting-label"><label for="custom-system-prompt">Custom instructions</label></div>
          <textarea class="setting-textarea" id="custom-system-prompt" maxlength="8000" placeholder="Example: Answer in plain language and include one practical example."></textarea>
          <div class="setting-desc">These instructions take priority over the selected persona.</div>
        </div>
        <div class="setting-row">
          <div class="setting-label">
            <label for="temperature-slider">Creativity (temperature)</label>
            <span id="temperature-value" style="font-family: monospace; color: #79c0ff;">0.7</span>
          </div>
          <input type="range" id="temperature-slider" min="0" max="2" step="0.1" value="0.7">
          <div class="setting-desc">Lower values are more consistent; higher values are more varied.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // State & Constants
    const MAX_CHATS = 50;
    let currentChatId = null;
    let chats = [];
    let streamAbortController = null;

    try {
      const stored = localStorage.getItem('router_chats');
      if (stored) chats = JSON.parse(stored);
      if (!Array.isArray(chats)) chats = [];
      if (chats.length > MAX_CHATS) {
        chats = chats.slice(0, MAX_CHATS);
      }
    } catch (e) {
      chats = [];
    }

    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const chatInner = document.getElementById('chat-inner');
    const heroSection = document.getElementById('hero-section');
    const chatScroll = document.getElementById('chat-scroll');
    const modelSelect = document.getElementById('model-select');
    const ctxSlider = document.getElementById('modal-ctx-slider');
    const ctxVal = document.getElementById('modal-ctx-val');
    const stripChk = document.getElementById('modal-strip-chk');
    const advancedModeChk = document.getElementById('advanced-mode-chk');
    const advancedPanel = document.getElementById('advanced-panel');
    const customSystemPrompt = document.getElementById('custom-system-prompt');
    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureValue = document.getElementById('temperature-value');
    const modelChoicesInput = document.getElementById('model-choices-input');
    const codingModelSelect = document.getElementById('coding-model-select');
    const reasoningModelSelect = document.getElementById('reasoning-model-select');
    const fallbackModelSelect = document.getElementById('fallback-model-select');
    const codingKeywordsInput = document.getElementById('coding-keywords-input');
    const reasoningKeywordsInput = document.getElementById('reasoning-keywords-input');
    const saveRoutingSettingsButton = document.getElementById('save-routing-settings');
    const routingSettingsStatus = document.getElementById('routing-settings-status');

    try {
      const savedAdvanced = localStorage.getItem('router_advanced_mode') === 'true';
      if (advancedModeChk) advancedModeChk.checked = savedAdvanced;
      if (advancedPanel) advancedPanel.classList.toggle('open', savedAdvanced);
      const savedSystemPrompt = localStorage.getItem('router_custom_system_prompt');
      if (savedSystemPrompt && customSystemPrompt) customSystemPrompt.value = savedSystemPrompt;
      const savedTemperature = localStorage.getItem('router_temperature');
      if (savedTemperature && temperatureSlider) temperatureSlider.value = savedTemperature;
      if (temperatureValue && temperatureSlider) temperatureValue.textContent = Number(temperatureSlider.value).toFixed(1);
    } catch (e) {}

    // 1. Attach Event Listeners FIRST so input is always responsive
    if (promptInput) {
      promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
        if (sendBtn) {
          if (promptInput.value.trim().length > 0) {
            sendBtn.classList.add('active');
            sendBtn.disabled = false;
          } else {
            sendBtn.classList.remove('active');
          }
        }
      });

      promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          submitMessage();
        }
      });
    }

    if (sendBtn) {
      sendBtn.onclick = () => submitMessage();
    }

    if (ctxSlider && ctxVal) {
      ctxSlider.addEventListener('input', (e) => {
        ctxVal.textContent = e.target.value;
      });
    }

    if (advancedModeChk && advancedPanel) {
      advancedModeChk.addEventListener('change', () => {
        advancedPanel.classList.toggle('open', advancedModeChk.checked);
        try { localStorage.setItem('router_advanced_mode', String(advancedModeChk.checked)); } catch (e) {}
      });
    }
    if (customSystemPrompt) {
      customSystemPrompt.addEventListener('input', () => {
        try { localStorage.setItem('router_custom_system_prompt', customSystemPrompt.value); } catch (e) {}
      });
    }
    if (temperatureSlider && temperatureValue) {
      temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = Number(temperatureSlider.value).toFixed(1);
        try { localStorage.setItem('router_temperature', temperatureSlider.value); } catch (e) {}
      });
    }

    function parseCommaList(value) {
      return String(value || '').split(',').map(item => item.trim()).filter(Boolean);
    }
    function populateSelect(select, values, selectedValue, includeAuto = false) {
      if (!select) return;
      select.textContent = '';
      if (includeAuto) {
        const autoOption = document.createElement('option');
        autoOption.value = 'auto';
        autoOption.textContent = '✨ Auto Router';
        select.appendChild(autoOption);
      }
      values.forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
      if (values.includes(selectedValue) || (includeAuto && selectedValue === 'auto')) {
        select.value = selectedValue;
      } else if (includeAuto) {
        select.value = 'auto';
      } else if (values.length) {
        select.value = values[0];
      }
    }
    function applyRouterSettings(settings) {
      const models = settings.model_choices || [];
      populateSelect(modelSelect, models, modelSelect ? modelSelect.value : 'auto', true);
      populateSelect(codingModelSelect, models, settings.coding_model);
      populateSelect(reasoningModelSelect, models, settings.reasoning_model);
      populateSelect(fallbackModelSelect, models, settings.fallback_model);
      if (modelChoicesInput) modelChoicesInput.value = models.join(', ');
      if (codingKeywordsInput) codingKeywordsInput.value = (settings.coding_keywords || []).join(', ');
      if (reasoningKeywordsInput) reasoningKeywordsInput.value = (settings.reasoning_keywords || []).join(', ');
    }
    async function loadRouterSettings() {
      try {
        const response = await fetch('/v1/settings');
        if (!response.ok) throw new Error('Could not load routing settings');
        applyRouterSettings(await response.json());
      } catch (error) {
        if (routingSettingsStatus) routingSettingsStatus.textContent = 'Could not load routing settings.';
      }
    }
    if (modelChoicesInput) {
      modelChoicesInput.addEventListener('input', () => {
        const models = parseCommaList(modelChoicesInput.value);
        populateSelect(codingModelSelect, models, codingModelSelect.value);
        populateSelect(reasoningModelSelect, models, reasoningModelSelect.value);
        populateSelect(fallbackModelSelect, models, fallbackModelSelect.value);
      });
    }
    if (saveRoutingSettingsButton) {
      saveRoutingSettingsButton.addEventListener('click', async () => {
        const settings = {
          model_choices: parseCommaList(modelChoicesInput ? modelChoicesInput.value : ''),
          coding_model: codingModelSelect ? codingModelSelect.value : '',
          reasoning_model: reasoningModelSelect ? reasoningModelSelect.value : '',
          fallback_model: fallbackModelSelect ? fallbackModelSelect.value : '',
          coding_keywords: parseCommaList(codingKeywordsInput ? codingKeywordsInput.value : ''),
          reasoning_keywords: parseCommaList(reasoningKeywordsInput ? reasoningKeywordsInput.value : '')
        };
        saveRoutingSettingsButton.disabled = true;
        if (routingSettingsStatus) routingSettingsStatus.textContent = 'Saving…';
        try {
          const response = await fetch('/v1/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
          });
          const result = await response.json();
          if (!response.ok) {
            const detail = Array.isArray(result.detail)
              ? result.detail.map(error => error.msg).join(', ')
              : (result.detail || 'Could not save settings');
            throw new Error(detail);
          }
          applyRouterSettings(result);
          if (routingSettingsStatus) routingSettingsStatus.textContent = 'Saved.';
        } catch (error) {
          if (routingSettingsStatus) routingSettingsStatus.textContent = error.message;
        } finally {
          saveRoutingSettingsButton.disabled = false;
        }
      });
    }

    // Sidebar & Modal Toggles
    function toggleSidebar() {
      const sb = document.getElementById('sidebar');
      if (sb) sb.classList.toggle('collapsed');
    }
    function openSettings() {
      const modal = document.getElementById('settings-modal');
      if (modal) modal.classList.add('open');
    }
    function closeSettings(e) {
      const modal = document.getElementById('settings-modal');
      if (modal) modal.classList.remove('open');
    }

    // Health check
    async function updateHealth() {
      try {
        const res = await fetch('/health');
        const data = await res.json();
        const dot = document.getElementById('sidebar-dot');
        const txt = document.getElementById('sidebar-status');
        if (dot && txt) {
          if (data.ollama_connected) {
            dot.className = 'status-dot online';
            txt.textContent = 'Ollama Connected';
          } else {
            dot.className = 'status-dot offline';
            txt.textContent = 'Ollama Offline';
          }
        }
      } catch (e) {
        const dot = document.getElementById('sidebar-dot');
        const txt = document.getElementById('sidebar-status');
        if (dot) dot.className = 'status-dot offline';
        if (txt) txt.textContent = 'Router Offline';
      }
    }
    loadRouterSettings();
    updateHealth();
    setInterval(updateHealth, 10000);

    // Chat Management
    function renderHistory() {
      const list = document.getElementById('history-list');
      if (!list) return;
      list.innerHTML = '';
      chats.forEach((c) => {
        if (!c) return;
        const item = document.createElement('div');
        item.className = 'history-item' + (c.id === currentChatId ? ' active' : '');
        item.innerHTML = `<span>💬 ${escapeHtml(c.title || 'Conversation')}</span>`;
        const deleteButton = document.createElement('span');
        deleteButton.className = 'del-chat';
        deleteButton.textContent = '×';
        deleteButton.addEventListener('click', (event) => deleteChat(event, c.id));
        item.appendChild(deleteButton);
        item.onclick = () => loadChat(c.id);
        list.appendChild(item);
      });
    }

    function stopGenerating() {
      if (streamAbortController) {
        streamAbortController.abort();
        streamAbortController = null;
      }
    }

    function startNewChat() {
      stopGenerating();
      currentChatId = 'chat_' + Date.now();
      chats.unshift({ id: currentChatId, title: 'New Conversation', messages: [] });
      saveChats();
      loadChat(currentChatId);
    }

    function loadChat(id) {
      if (currentChatId !== id) {
        stopGenerating();
      }
      currentChatId = id;
      const chat = chats.find(c => c && c.id === id);
      if (!chatInner) return;
      chatInner.innerHTML = '';

      if (!chat || !Array.isArray(chat.messages) || chat.messages.length === 0) {
        if (heroSection) {
          chatInner.appendChild(heroSection);
          heroSection.style.display = 'flex';
        }
      } else {
        if (heroSection) heroSection.style.display = 'none';
        chat.messages.forEach(m => {
          if (m && m.content) renderMessageDOM(m);
        });
      }
      renderHistory();
      if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function deleteChat(e, id) {
      if (e) e.stopPropagation();
      if (currentChatId === id) {
        stopGenerating();
      }
      chats = chats.filter(c => c && c.id !== id);
      saveChats();
      if (currentChatId === id) {
        if (chats.length > 0) loadChat(chats[0].id);
        else startNewChat();
      } else {
        renderHistory();
      }
    }

    function saveChats() {
      try {
        if (chats.length > MAX_CHATS) {
          chats = chats.slice(0, MAX_CHATS);
        }
        localStorage.setItem('router_chats', JSON.stringify(chats));
      } catch (e) {
        console.warn('localStorage quota warning or error:', e);
        try {
          while (chats.length > 10) {
            chats.pop();
            try {
              localStorage.setItem('router_chats', JSON.stringify(chats));
              break;
            } catch (innerErr) {}
          }
        } catch (ignored) {}
      }
    }

    function fillAndSend(text) {
      if (!promptInput) return;
      promptInput.value = text;
      promptInput.dispatchEvent(new Event('input'));
      submitMessage();
    }

    // Send Message
    async function submitMessage() {
      if (!promptInput) return;
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      if (!currentChatId) {
        currentChatId = 'chat_' + Date.now();
        chats.unshift({ id: currentChatId, title: prompt.slice(0, 30), messages: [] });
      }

      const chat = chats.find(c => c && c.id === currentChatId);
      if (chat && (!chat.messages || chat.messages.length === 0)) {
        chat.title = prompt.slice(0, 30) + (prompt.length > 30 ? '...' : '');
      }
      const history = (chat && Array.isArray(chat.messages) ? chat.messages : [])
        .filter(message => message && (message.role === 'user' || message.role === 'bot') && message.content)
        .slice(-12)
        .map(message => ({
          role: message.role === 'user' ? 'user' : 'assistant',
          content: message.content
        }));

      if (heroSection) heroSection.style.display = 'none';

      // 1. User Message
      const userMsg = { role: 'user', content: prompt };
      if (chat) {
        if (!chat.messages) chat.messages = [];
        chat.messages.push(userMsg);
      }
      renderMessageDOM(userMsg);

      promptInput.value = '';
      promptInput.style.height = '28px';
      if (sendBtn) {
        sendBtn.classList.remove('active');
        sendBtn.disabled = true;
      }

      // 2. Bot Placeholder
      const botMsgId = 'bot_' + Date.now();
      const placeholder = document.createElement('div');
      placeholder.className = 'msg-row';
      placeholder.id = botMsgId;
      placeholder.innerHTML = `
        <div class="avatar">✦</div>
        <div class="msg-bot-content">
          <div class="bot-header-badge">⚡ Classifying intent & generating...</div>
          <div style="color: #888; font-style: italic;">Processing prompt on local Ollama...</div>
        </div>
      `;
      if (chatInner) chatInner.appendChild(placeholder);
      if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;

      // 3. API Call with Streaming & Persona
      const model = modelSelect ? modelSelect.value : 'auto';
      const personaEl = document.getElementById('persona-select');
      const persona = personaEl ? personaEl.value : 'general';
      const contextLimit = ctxSlider ? parseInt(ctxSlider.value, 10) : 2048;
      const stripReasoning = stripChk ? stripChk.checked : true;
      const advancedMode = advancedModeChk ? advancedModeChk.checked : false;

      stopGenerating();
      streamAbortController = new AbortController();
      if (stopBtn) stopBtn.style.display = 'inline-flex';

      try {
        const res = await fetch('/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: streamAbortController.signal,
          body: JSON.stringify({
            prompt: prompt,
            history: history,
            model: model,
            persona: persona,
            context_limit: contextLimit,
            strip_reasoning: stripReasoning,
            ...(advancedMode ? {
              system_prompt: customSystemPrompt ? customSystemPrompt.value.trim() : '',
              temperature: temperatureSlider ? Number(temperatureSlider.value) : 0.7
            } : {}),
            stream: true
          })
        });

        if (!res.ok) {
          let errorDetail = 'Request failed (' + res.status + ')';
          try {
            const err = await res.json();
            errorDetail = err.detail || JSON.stringify(err);
          } catch (e) {
            const textErr = await res.text();
            if (textErr) errorDetail = textErr;
          }
          placeholder.remove();
          const errPayload = { role: 'bot', content: '❌ Error: ' + errorDetail };
          if (chat) chat.messages.push(errPayload);
          renderMessageDOM(errPayload);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let fullResponse = '';
        let targetModel = model === 'auto' ? 'routing...' : model;
        let evalCount = 0;
        let thinkContent = '';
        let isInsideThink = false;
        let buffer = '';

        const botMsgContent = placeholder.querySelector('.msg-bot-content');

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep remainder

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            try {
              const chunk = JSON.parse(dataStr);
              if (chunk.model) targetModel = chunk.model;
              if (chunk.eval_count) evalCount = chunk.eval_count;
              if (chunk.response) {
                const token = chunk.response;
                if (token.includes('<think>')) isInsideThink = true;
                if (isInsideThink) {
                  thinkContent += token;
                  if (token.includes('</think>')) isInsideThink = false;
                } else {
                  fullResponse += token;
                }

                // Render live typewriter
                const cleanThink = thinkContent.replace(/<\/?think>/g, '').trim();
                const thinkHtml = cleanThink ? `<details class="think-box" open><summary>🧠 Thinking Process...</summary><pre>${escapeHtml(cleanThink)}</pre></details>` : '';
                
                const renderedMd = renderMarkdownSafe(fullResponse || '');

                if (botMsgContent) {
                  botMsgContent.innerHTML = `
                    <div class="bot-header-badge">⚡ ${escapeHtml(targetModel)} • ${escapeHtml(persona)}</div>
                    ${thinkHtml}
                    <div class="markdown-body">${renderedMd}</div>
                  `;
                }
                if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
              }
            } catch (e) {}
          }
        }

        // Clean thinking tags for final history persistence
        let finalCleanReasoning = null;
        if (thinkContent) {
          finalCleanReasoning = thinkContent.replace(/<\/?think>/g, '').trim();
        }

        placeholder.remove();
        const botMsg = {
          role: 'bot',
          content: fullResponse || '(Empty response)',
          model: targetModel,
          reasoning: finalCleanReasoning,
          tokens: { eval_count: evalCount }
        };
        if (chat) chat.messages.push(botMsg);
        renderMessageDOM(botMsg);
        saveChats();
        renderHistory();

      } catch (err) {
        placeholder.remove();
        if (err.name === 'AbortError') {
          let finalCleanReasoning = null;
          if (thinkContent) {
            finalCleanReasoning = thinkContent.replace(/<\/?think>/g, '').trim();
          }
          const botMsg = {
            role: 'bot',
            content: fullResponse ? (fullResponse + ' *(generation stopped)*') : '*(Generation stopped by user)*',
            model: targetModel,
            reasoning: finalCleanReasoning,
            tokens: { eval_count: evalCount }
          };
          if (chat) chat.messages.push(botMsg);
          renderMessageDOM(botMsg);
          saveChats();
          renderHistory();
        } else {
          const errPayload = { role: 'bot', content: '❌ Network Error: ' + err.message };
          if (chat) chat.messages.push(errPayload);
          renderMessageDOM(errPayload);
        }
      } finally {
        streamAbortController = null;
        if (stopBtn) stopBtn.style.display = 'none';
        if (sendBtn) {
          sendBtn.disabled = false;
          if (promptInput && promptInput.value.trim().length > 0) {
            sendBtn.classList.add('active');
          } else {
            sendBtn.classList.remove('active');
          }
        }
        if (promptInput) promptInput.focus();
      }
    }

    function renderMessageDOM(msg) {
      if (!chatInner || !msg) return;
      const row = document.createElement('div');
      row.className = 'msg-row ' + (msg.role || 'bot');

      if (msg.role === 'user') {
        row.innerHTML = `<div class="msg-user-bubble">${escapeHtml(msg.content)}</div>`;
      } else {
        let metaBadge = '';
        if (msg.model) {
          const tokenStr = msg.tokens && msg.tokens.eval_count ? ` • ${escapeHtml(msg.tokens.eval_count)} tokens` : '';
          metaBadge = `<div class="bot-header-badge">⚡ ${escapeHtml(msg.model)}${tokenStr}</div>`;
        }

        let thinkHtml = '';
        if (msg.reasoning) {
          thinkHtml = `<details class="think-box"><summary>🧠 View Thinking Process</summary><pre>${escapeHtml(msg.reasoning)}</pre></details>`;
        }

        const renderedMd = renderMarkdownSafe(msg.content || '');

        row.innerHTML = `
          <div class="avatar">✦</div>
          <div class="msg-bot-content">
            ${metaBadge}
            ${thinkHtml}
            <div class="markdown-body">${renderedMd}</div>
          </div>
        `;

        // Highlight code & inject copy buttons safely
        if (typeof hljs !== 'undefined' && typeof hljs.highlightElement === 'function') {
          row.querySelectorAll('pre code').forEach((block) => {
            try {
              hljs.highlightElement(block);
              const pre = block.parentElement;
              const lang = block.className.replace('hljs language-', '').replace('hljs', '').trim() || 'code';
              const header = document.createElement('div');
              header.className = 'code-header';
              header.innerHTML = `<span>${lang}</span><button class="copy-btn" onclick="copyCode(this)">📋 Copy</button>`;
              pre.insertBefore(header, block);
            } catch (e) {}
          });
        }
      }

      chatInner.appendChild(row);
      if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function copyCode(btn) {
      const pre = btn.closest('pre');
      const code = pre.querySelector('code').innerText;
      navigator.clipboard.writeText(code).then(() => {
        btn.innerHTML = '✓ Copied!';
        setTimeout(() => { btn.innerHTML = '📋 Copy'; }, 2000);
      });
    }

    function escapeHtml(str) {
      return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function renderMarkdownSafe(source) {
      if (typeof marked === 'undefined' || typeof marked.parse !== 'function') {
        return escapeHtml(source);
      }
      try {
        const parsed = new DOMParser().parseFromString(marked.parse(source), 'text/html');
        const allowedTags = new Set([
          'p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
          'ul', 'ol', 'li', 'strong', 'em', 'del', 'a', 'code', 'pre',
          'table', 'thead', 'tbody', 'tr', 'th', 'td'
        ]);
        const dropTags = new Set(['script', 'style', 'iframe', 'object', 'embed', 'svg', 'math', 'form']);
        const safeUrl = (value) => {
          try {
            const url = new URL(value, window.location.href);
            return ['http:', 'https:', 'mailto:'].includes(url.protocol) ? value : null;
          } catch (e) { return null; }
        };
        const cleanNode = (node) => {
          if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.nodeValue || '');
          if (node.nodeType !== Node.ELEMENT_NODE) return document.createDocumentFragment();
          const tag = node.tagName.toLowerCase();
          if (dropTags.has(tag)) return document.createDocumentFragment();
          if (!allowedTags.has(tag)) {
            const fragment = document.createDocumentFragment();
            Array.from(node.childNodes).forEach(child => fragment.appendChild(cleanNode(child)));
            return fragment;
          }
          const clean = document.createElement(tag);
          if (tag === 'a') {
            const href = node.getAttribute('href');
            const safeHref = href ? safeUrl(href) : null;
            if (safeHref) clean.setAttribute('href', safeHref);
            const title = node.getAttribute('title');
            if (title) clean.setAttribute('title', title);
          }
          if (tag === 'code' || tag === 'th' || tag === 'td') {
            const className = node.getAttribute('class') || '';
            if (/^(language-[a-zA-Z0-9_+-]+|hljs)(\s+(language-[a-zA-Z0-9_+-]+|hljs))*$/.test(className)) {
              clean.setAttribute('class', className);
            }
          }
          Array.from(node.childNodes).forEach(child => clean.appendChild(cleanNode(child)));
          return clean;
        };
        const safeRoot = document.createElement('div');
        Array.from(parsed.body.childNodes).forEach(node => safeRoot.appendChild(cleanNode(node)));
        return safeRoot.innerHTML;
      } catch (e) {
        return escapeHtml(source);
      }
    }

    // Initial load wrapped in safety catch
    try {
      if (chats.length === 0) {
        startNewChat();
      } else {
        loadChat(chats[0].id);
      }
    } catch (e) {
      console.warn("Recovered from stored chat parse error:", e);
      startNewChat();
    }
  </script>
</body>
</html>
"""


# ---- GET / ----------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive Web UI landing page."""
    return HTMLResponse(content=INDEX_HTML)


# ---- Helpers for Personas & Streaming -----------------------------------


def resolve_persona_config(
    persona_key: Optional[str],
    custom_system: Optional[str],
    custom_temp: Optional[float],
) -> tuple[Optional[str], Optional[float]]:
    """Resolve system prompt and temperature from persona presets and overrides."""
    system = None
    temp = None
    if persona_key and persona_key in PERSONAS:
        system = PERSONAS[persona_key]["system"]
        temp = PERSONAS[persona_key]["temperature"]
    if custom_system:
        system = custom_system
    if custom_temp is not None:
        temp = custom_temp
    return system, temp


async def stream_chat_generator(
    client: httpx.AsyncClient, payload: dict, target_model: str
):
    """Stream token chunks from Ollama as Server-Sent Events (SSE)."""
    try:
        async with client.stream("POST", OLLAMA_GENERATE, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    yield f"data: {json.dumps(chunk)}\n\n"
                except Exception:
                    continue
            yield "data: [DONE]\n\n"
    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Ollama server offline or unreachable'})}\n\n"
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'error': 'Upstream request timed out'})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


async def stream_openai_generator(
    client: httpx.AsyncClient, payload: dict, target_model: str, completion_id: str
):
    """Stream OpenAI-compatible chat completion chunks."""
    created_time = int(time.time())
    try:
        async with client.stream("POST", OLLAMA_GENERATE, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    done = chunk.get("done", False)

                    event = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": target_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token} if not done else {},
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception:
                    continue
            yield "data: [DONE]\n\n"
    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Ollama server offline or unreachable'})}\n\n"
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'error': 'Upstream request timed out'})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


# ---- POST /v1/chat -------------------------------------------------------


@app.post("/v1/chat")
async def chat(
    request: ChatRequest,
    client: httpx.AsyncClient = Depends(get_http_client),
):
    """Send a prompt to Ollama with auto-routing, persona injection, and optional streaming."""

    # 1. Resolve target model
    if request.model == "auto":
        target_model = classify_prompt(request.prompt)
    else:
        if request.model not in ROUTER_SETTINGS.model_choices:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. "
                f"Available: {ROUTER_SETTINGS.model_choices}",
            )
        target_model = request.model

    # 2. Resolve persona system prompt and temperature
    system_prompt, temp = resolve_persona_config(
        request.persona, request.system_prompt, request.temperature
    )

    # 3. Build Ollama payload (enforce num_ctx for VRAM safety)
    options: dict = {"num_ctx": request.context_limit}
    if temp is not None:
        options["temperature"] = temp
    if target_model == "qwen2.5:3b":
        options["num_gpu"] = 0  # 3B params exceed 4GB VRAM with desktop overhead; offload safely to CPU

    payload = {
        "model": target_model,
        "prompt": format_prompt_with_history(request.prompt, request.history),
        "stream": request.stream,
        "options": options,
    }
    if system_prompt:
        payload["system"] = system_prompt

    # 4. Handle streaming mode
    if request.stream:
        return StreamingResponse(
            stream_chat_generator(client, payload, target_model),
            media_type="text/event-stream",
        )

    # 5. Non-streaming request
    try:
        resp = await client.post(OLLAMA_GENERATE, json=payload)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama server is offline or unreachable at "
            f"{OLLAMA_BASE}. Please start Ollama and try again.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Inference request timed out after {OLLAMA_TIMEOUT}s.",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {exc.response.status_code} — "
            f"{exc.response.text}",
        )

    data = resp.json()
    response_text: str = data.get("response", "")

    # 6. Process <think> tags for deepseek-r1
    reasoning = None
    if target_model == "deepseek-r1:1.5b":
        response_text, reasoning = process_reasoning(
            response_text, request.strip_reasoning
        )

    return {
        "model": target_model,
        "persona": request.persona,
        "response": response_text,
        "reasoning": reasoning,
        "tokens": {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        },
    }


# ---- POST /v1/chat/completions (OpenAI Compatible) -----------------------


@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    client: httpx.AsyncClient = Depends(get_http_client),
):
    """OpenAI-compatible chat completion endpoint.

    Allows tools like Continue.dev, Cursor, Obsidian, or official OpenAI SDK
    to treat this local router as a drop-in OpenAI replacement.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    # Extract user prompt and conversation history
    system_instruction = None
    conversation_lines = []
    last_user_prompt = ""

    for msg in request.messages:
        if msg.role == "system":
            system_instruction = msg.content
        elif msg.role == "user":
            conversation_lines.append(f"User: {msg.content}")
            last_user_prompt = msg.content
        elif msg.role == "assistant":
            conversation_lines.append(f"Assistant: {msg.content}")

    # Determine target model (auto-route using the last user prompt if auto)
    if not request.model or request.model == "auto":
        target_model = classify_prompt(last_user_prompt)
    elif request.model in ROUTER_SETTINGS.model_choices:
        target_model = request.model
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model}. Available: {ROUTER_SETTINGS.model_choices} or 'auto'.",
        )

    # Resolve persona
    persona_system, persona_temp = resolve_persona_config(
        request.persona, system_instruction, request.temperature
    )

    full_prompt = "\n".join(conversation_lines)
    options: dict = {"num_ctx": request.context_limit or 2048}
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if persona_temp is not None:
        options["temperature"] = persona_temp
    if target_model == "qwen2.5:3b":
        options["num_gpu"] = 0

    payload = {
        "model": target_model,
        "prompt": full_prompt,
        "stream": bool(request.stream),
        "options": options,
    }
    if persona_system:
        payload["system"] = persona_system

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Streaming mode
    if request.stream:
        return StreamingResponse(
            stream_openai_generator(client, payload, target_model, completion_id),
            media_type="text/event-stream",
        )

    # Non-streaming mode
    try:
        resp = await client.post(OLLAMA_GENERATE, json=payload)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama server is offline or unreachable at "
            f"{OLLAMA_BASE}. Please start Ollama and try again.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Inference request timed out after {OLLAMA_TIMEOUT}s.",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {exc.response.status_code} — "
            f"{exc.response.text}",
        )

    data = resp.json()
    response_text = data.get("response", "")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": target_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": (data.get("prompt_eval_count", 0) or 0)
            + (data.get("eval_count", 0) or 0),
        },
    }


# ---- GET /v1/personas ----------------------------------------------------


@app.get("/v1/personas")
async def list_personas():
    """Return available persona presets and their system prompt configurations."""
    return {"personas": PERSONAS}


# ---- GET/PUT /v1/settings -----------------------------------------------


@app.get("/v1/settings")
async def get_router_settings():
    """Return editable local routing rules and configured model tags."""
    return ROUTER_SETTINGS.model_dump()


@app.put("/v1/settings")
async def update_router_settings(settings: RouterSettings):
    """Validate and persist routing rules and model choices."""
    global ROUTER_SETTINGS
    try:
        save_router_settings(settings)
    except OSError:
        logging.getLogger(__name__).exception("Could not save router settings")
        raise HTTPException(status_code=500, detail="Could not save router settings")
    ROUTER_SETTINGS = settings
    return ROUTER_SETTINGS.model_dump()


# ---- GET /v1/models ------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    """Return configured models and the active auto-routing rules."""
    return {
        "installed_models": ROUTER_SETTINGS.model_choices,
        "routing_options": ["auto", *ROUTER_SETTINGS.model_choices],
        "auto_routing_rules": {
            "coding_keywords": ", ".join(ROUTER_SETTINGS.coding_keywords),
            "reasoning_keywords": ", ".join(ROUTER_SETTINGS.reasoning_keywords),
            "coding_model": ROUTER_SETTINGS.coding_model,
            "reasoning_model": ROUTER_SETTINGS.reasoning_model,
            "default_fallback": ROUTER_SETTINGS.fallback_model,
        },
    }


# ---- GET /health ----------------------------------------------------------


@app.get("/health")
async def health(client: httpx.AsyncClient = Depends(get_http_client)):
    """Check service health and Ollama connectivity."""
    ollama_ok = False
    try:
        resp = await client.get(f"{OLLAMA_BASE}/", timeout=5.0)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass
    return JSONResponse(
        status_code=200 if ollama_ok else 503,
        content={
            "status": "healthy" if ollama_ok else "unhealthy",
            "ollama_connected": ollama_ok,
        },
    )
