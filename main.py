"""
FastAPI Ollama Model Router
----------------------------
Lightweight orchestration layer for a local Ollama server.
Auto-routes prompts to the best model via keyword heuristics,
enforces VRAM-safe context limits, and strips DeepSeek <think> tags.
"""

import json
import re
import time
import uuid
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT = f"{OLLAMA_BASE}/api/chat"
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
# Pydantic request models
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
    persona: Optional[Literal["general", "coder", "reasoner", "executive", "custom"]] = "general"
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    stream: bool = False
    strip_reasoning: bool = True
    context_limit: int = Field(default=2048, ge=512, le=4096)


class ChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = "auto"
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    context_limit: Optional[int] = Field(default=2048, ge=512, le=4096)
    persona: Optional[str] = "general"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def classify_prompt(prompt: str) -> str:
    """Route a prompt to the best model using keyword heuristics.

    Priority: Coding → Reasoning → Ultra-fast fallback (gemma3:1b).
    """
    if CODING_PATTERN.search(prompt):
        return "qwen2.5-coder:1.5b-base"
    if REASONING_PATTERN.search(prompt):
        return "deepseek-r1:1.5b"
    return "gemma3:1b"


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


# ---------------------------------------------------------------------------
# Interactive Web UI
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
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
      max-width: 440px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modal-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    .modal-close { background: transparent; border: none; color: #aaa; cursor: pointer; font-size: 18px; }
    .setting-row { margin-bottom: 16px; }
    .setting-label { font-size: 0.88rem; font-weight: 500; margin-bottom: 6px; display: flex; justify-content: space-between; }
    .setting-desc { font-size: 0.75rem; color: #999; margin-top: 4px; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
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
    </div>
  </div>

  <script>
    // State
    let currentChatId = null;
    let chats = [];
    try {
      const stored = localStorage.getItem('router_chats');
      if (stored) chats = JSON.parse(stored);
      if (!Array.isArray(chats)) chats = [];
    } catch (e) {
      chats = [];
    }

    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const chatInner = document.getElementById('chat-inner');
    const heroSection = document.getElementById('hero-section');
    const chatScroll = document.getElementById('chat-scroll');
    const modelSelect = document.getElementById('model-select');
    const ctxSlider = document.getElementById('modal-ctx-slider');
    const ctxVal = document.getElementById('modal-ctx-val');
    const stripChk = document.getElementById('modal-strip-chk');

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
        item.innerHTML = `<span>💬 ${escapeHtml(c.title || 'Conversation')}</span><span class="del-chat" onclick="deleteChat(event, '${c.id}')">&times;</span>`;
        item.onclick = () => loadChat(c.id);
        list.appendChild(item);
      });
    }

    function startNewChat() {
      currentChatId = 'chat_' + Date.now();
      chats.unshift({ id: currentChatId, title: 'New Conversation', messages: [] });
      saveChats();
      loadChat(currentChatId);
    }

    function loadChat(id) {
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
        localStorage.setItem('router_chats', JSON.stringify(chats));
      } catch (e) {}
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

      try {
        const res = await fetch('/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            model: model,
            persona: persona,
            context_limit: contextLimit,
            strip_reasoning: stripReasoning,
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
                
                let renderedMd = escapeHtml(fullResponse || '');
                if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
                  try { renderedMd = marked.parse(fullResponse || ''); } catch (e) {}
                }

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
        const errPayload = { role: 'bot', content: '❌ Network Error: ' + err.message };
        if (chat) chat.messages.push(errPayload);
        renderMessageDOM(errPayload);
      } finally {
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.classList.remove('active');
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
          const tokenStr = msg.tokens && msg.tokens.eval_count ? ` • ${msg.tokens.eval_count} tokens` : '';
          metaBadge = `<div class="bot-header-badge">⚡ ${escapeHtml(msg.model)}${tokenStr}</div>`;
        }

        let thinkHtml = '';
        if (msg.reasoning) {
          thinkHtml = `<details class="think-box"><summary>🧠 View Thinking Process</summary><pre>${escapeHtml(msg.reasoning)}</pre></details>`;
        }

        // Render Markdown safely
        let renderedMd = escapeHtml(msg.content || '');
        if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
          try {
            renderedMd = marked.parse(msg.content || '');
          } catch (e) {
            renderedMd = escapeHtml(msg.content || '');
          }
        }

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


async def stream_chat_generator(payload: dict, target_model: str):
    """Stream token chunks from Ollama as Server-Sent Events (SSE)."""
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
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
        yield f"data: {json.dumps({'error': 'Ollama server offline'})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


async def stream_openai_generator(payload: dict, target_model: str, completion_id: str):
    """Stream OpenAI-compatible chat completion chunks."""
    created_time = int(time.time())
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
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
        yield f"data: {json.dumps({'error': 'Ollama server offline'})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


# ---- POST /v1/chat -------------------------------------------------------


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    """Send a prompt to Ollama with auto-routing, persona injection, and optional streaming."""

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
        "prompt": request.prompt,
        "stream": request.stream,
        "options": options,
    }
    if system_prompt:
        payload["system"] = system_prompt

    # 4. Handle streaming mode
    if request.stream:
        return StreamingResponse(
            stream_chat_generator(payload, target_model),
            media_type="text/event-stream",
        )

    # 5. Non-streaming request
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
async def chat_completions(request: OpenAIChatCompletionRequest):
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
    elif request.model in INSTALLED_MODELS:
        target_model = request.model
    else:
        target_model = "qwen2.5:3b"  # fallback

    # Resolve persona
    persona_system, persona_temp = resolve_persona_config(
        request.persona, system_instruction, request.temperature
    )

    full_prompt = "\n".join(conversation_lines)
    options: dict = {"num_ctx": request.context_limit or 2048}
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
            stream_openai_generator(payload, target_model, completion_id),
            media_type="text/event-stream",
        )

    # Non-streaming mode
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(OLLAMA_GENERATE, json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama server offline")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text}")

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
