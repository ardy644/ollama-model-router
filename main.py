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
from fastapi.responses import HTMLResponse
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


# ---------------------------------------------------------------------------
# Interactive Web UI
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ollama Model Router</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #0d1117; color: #c9d1d9; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 1.1rem; color: #f0f6fc; }
    .badge-status { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; background: #21262d; border: 1px solid #30363d; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #8b949e; }
    .dot.online { background: #3fb950; box-shadow: 0 0 8px #3fb95088; }
    .dot.offline { background: #f85149; }
    .controls { background: #161b22; border-bottom: 1px solid #21262d; padding: 10px 24px; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; font-size: 0.85rem; }
    .ctrl-item { display: flex; align-items: center; gap: 8px; }
    select, input[type="range"] { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px; font-size: 0.85rem; outline: none; }
    select:focus { border-color: #58a6ff; }
    a.doc-link { color: #58a6ff; text-decoration: none; font-size: 0.85rem; margin-left: auto; }
    a.doc-link:hover { text-decoration: underline; }
    #chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
    .message { max-width: 80%; display: flex; flex-direction: column; gap: 6px; line-height: 1.5; }
    .message.user { align-self: flex-end; }
    .message.user .bubble { background: #1f6feb; color: #fff; border-radius: 14px 14px 2px 14px; padding: 12px 16px; }
    .message.bot { align-self: flex-start; max-width: 85%; }
    .message.bot .bubble { background: #161b22; border: 1px solid #30363d; border-radius: 14px 14px 14px 2px; padding: 14px 18px; word-break: break-word; }
    .meta-tag { font-size: 0.72rem; color: #8b949e; display: inline-flex; align-items: center; gap: 8px; margin-bottom: 4px; }
    .model-pill { background: #21262d; border: 1px solid #388bfd44; color: #58a6ff; padding: 2px 8px; border-radius: 10px; font-family: monospace; font-size: 0.75rem; }
    details.reasoning-box { background: #0d1117; border: 1px dashed #30363d; border-radius: 8px; padding: 8px 12px; margin-bottom: 10px; font-size: 0.82rem; color: #8b949e; }
    details.reasoning-box summary { cursor: pointer; color: #a5d6ff; font-weight: 500; }
    details.reasoning-box pre { margin-top: 8px; white-space: pre-wrap; font-family: inherit; }
    .input-bar { background: #161b22; border-top: 1px solid #30363d; padding: 16px 24px; display: flex; gap: 12px; }
    textarea { flex: 1; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; font-size: 0.95rem; resize: none; height: 50px; outline: none; }
    textarea:focus { border-color: #58a6ff; }
    button.send-btn { background: #238636; color: white; border: none; border-radius: 8px; padding: 0 22px; font-weight: 600; cursor: pointer; font-size: 0.95rem; transition: background 0.15s; }
    button.send-btn:hover { background: #2ea043; }
    button.send-btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
    .welcome-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; max-width: 600px; margin: auto; text-align: center; }
    .quick-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 14px; }
    .chip { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 14px; font-size: 0.8rem; cursor: pointer; transition: all 0.15s; }
    .chip:hover { border-color: #58a6ff; color: #58a6ff; }
    pre code { background: #0d1117; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>⚡ Ollama Model Router</span>
      <span class="badge-status">
        <span class="dot" id="status-dot"></span>
        <span id="status-text">Checking Ollama...</span>
      </span>
    </div>
    <a href="/docs" target="_blank" class="doc-link">Swagger API Docs ↗</a>
  </header>

  <div class="controls">
    <div class="ctrl-item">
      <label for="model-select"><strong>Model:</strong></label>
      <select id="model-select">
        <option value="auto">Auto (Smart Keyword Router)</option>
        <option value="qwen2.5-coder:1.5b-base">qwen2.5-coder:1.5b-base (Code)</option>
        <option value="deepseek-r1:1.5b">deepseek-r1:1.5b (Reasoning)</option>
        <option value="qwen2.5:3b">qwen2.5:3b (General)</option>
        <option value="gemma3:1b">gemma3:1b (Fast Fallback)</option>
      </select>
    </div>

    <div class="ctrl-item">
      <label for="ctx-slider">Context Window:</label>
      <input type="range" id="ctx-slider" min="512" max="4096" step="256" value="2048">
      <span id="ctx-val" style="font-family: monospace;">2048</span>
    </div>

    <div class="ctrl-item">
      <input type="checkbox" id="strip-chk" checked>
      <label for="strip-chk">Extract Reasoning (&lt;think&gt;)</label>
    </div>
  </div>

  <div id="chat-container">
    <div class="welcome-card" id="welcome-card">
      <h3 style="color: #f0f6fc; margin-bottom: 8px;">Local AI Gateway Ready</h3>
      <p style="font-size: 0.9rem; color: #8b949e;">Type a prompt below or pick a sample to see intent-based auto-routing in action:</p>
      <div class="quick-chips">
        <div class="chip" onclick="fillPrompt('Write a Python function to check if a word is palindrome')">💻 Python Function (Coder)</div>
        <div class="chip" onclick="fillPrompt('Solve step-by-step: A train leaves station A at 60 km/h...')">🧠 Math & Logic (DeepSeek)</div>
        <div class="chip" onclick="fillPrompt('Summarize why running LLMs locally protects privacy')">💬 General Prompt (Qwen 3B)</div>
        <div class="chip" onclick="fillPrompt('Say hi in Spanish')">⚡ Quick Greeting</div>
      </div>
    </div>
  </div>

  <div class="input-bar">
    <textarea id="prompt-input" placeholder="Type your prompt here... (Press Enter to send, Shift+Enter for newline)"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendPrompt()">Send</button>
  </div>

  <script>
    const ctxSlider = document.getElementById('ctx-slider');
    const ctxVal = document.getElementById('ctx-val');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const welcomeCard = document.getElementById('welcome-card');

    ctxSlider.addEventListener('input', (e) => { ctxVal.textContent = e.target.value; });

    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
      }
    });

    async function checkHealth() {
      try {
        const res = await fetch('/health');
        const data = await res.json();
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        if (data.ollama_connected) {
          dot.className = 'dot online';
          txt.textContent = 'Ollama Connected (GTX 1650)';
        } else {
          dot.className = 'dot offline';
          txt.textContent = 'Ollama Offline';
        }
      } catch (e) {
        document.getElementById('status-dot').className = 'dot offline';
        document.getElementById('status-text').textContent = 'Router Error';
      }
    }
    checkHealth();
    setInterval(checkHealth, 10000);

    function fillPrompt(text) {
      promptInput.value = text;
      promptInput.focus();
    }

    async function sendPrompt() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      if (welcomeCard) welcomeCard.style.display = 'none';

      // Append user bubble
      appendMessage('user', prompt);
      promptInput.value = '';
      sendBtn.disabled = true;
      sendBtn.textContent = 'Generating...';

      // Append temporary bot bubble
      const botMsgId = 'msg-' + Date.now();
      appendBotSkeleton(botMsgId);

      const model = document.getElementById('model-select').value;
      const contextLimit = parseInt(ctxSlider.value, 10);
      const stripReasoning = document.getElementById('strip-chk').checked;

      try {
        const res = await fetch('/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            model: model,
            context_limit: contextLimit,
            strip_reasoning: stripReasoning
          })
        });

        if (!res.ok) {
          const err = await res.json();
          updateBotMessage(botMsgId, 'Error: ' + (err.detail || 'Failed request'), null, null);
          return;
        }

        const data = await res.json();
        updateBotMessage(botMsgId, data.response, data.model, data.reasoning, data.tokens);
      } catch (err) {
        updateBotMessage(botMsgId, 'Error connecting to router: ' + err.message, null, null);
      } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
        promptInput.focus();
      }
    }

    function appendMessage(role, text) {
      const msg = document.createElement('div');
      msg.className = 'message ' + role;
      msg.innerHTML = '<div class="bubble">' + escapeHtml(text) + '</div>';
      chatContainer.appendChild(msg);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function appendBotSkeleton(id) {
      const msg = document.createElement('div');
      msg.className = 'message bot';
      msg.id = id;
      msg.innerHTML = '<div class="bubble" style="color: #8b949e;">Routing & generating response...</div>';
      chatContainer.appendChild(msg);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function updateBotMessage(id, text, model, reasoning, tokens) {
      const msg = document.getElementById(id);
      if (!msg) return;

      let metaHtml = '';
      if (model) {
        metaHtml = '<div class="meta-tag">' +
          '<span class="model-pill">⚡ ' + escapeHtml(model) + '</span>' +
          (tokens && tokens.eval_count ? '<span>' + tokens.eval_count + ' tokens</span>' : '') +
          '</div>';
      }

      let reasoningHtml = '';
      if (reasoning) {
        reasoningHtml = '<details class="reasoning-box"><summary>🧠 View Chain-of-Thought</summary><pre>' + escapeHtml(reasoning) + '</pre></details>';
      }

      msg.innerHTML = metaHtml +
        '<div class="bubble">' + reasoningHtml + '<div style="white-space: pre-wrap;">' + escapeHtml(text) + '</div></div>';
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(str) {
      return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
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
