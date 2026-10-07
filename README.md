# 🚀 Ollama Model Router

A lightweight, production-ready **FastAPI** orchestration layer that sits in front of a local [Ollama](https://ollama.com) server. It intelligently auto-routes prompts to the best local LLM based on keyword heuristics, enforces VRAM-safe context limits, and strips DeepSeek `<think>` reasoning tags.

Built for **constrained hardware** — designed to run comfortably on a 4 GB VRAM GPU.

---

## ✨ Features

- **ChatGPT & Gemini-Grade Web UI** — Responsive dark-mode interface with collapsible sidebar, multi-chat local history, markdown rendering, syntax-highlighted code with copy buttons, and collapsible thinking process drawers
- **Model Customization & Personas** — Pre-configured system prompt presets (`Senior Coder`, `Math & Logic Tutor`, `Executive Summarizer`, `General Assistant`) with tailored temperatures and formatting constraints
- **OpenAI-Compatible Endpoint (`/v1/chat/completions`)** — Drop-in replacement for OpenAI API; easily connect VS Code (Continue.dev), Obsidian, or standard OpenAI SDKs
- **Real-Time Token Streaming** — Word-by-word streaming using Server-Sent Events (SSE) for both the Web UI and API clients
- **Conversation Context** — The Web UI includes recent turns so follow-up questions stay on topic, with a bounded history sent to the model
- **Editable Routing** — Change routing keywords, fallback behavior, and available model choices in Settings
- **Automatic Ollama Startup** — Starts `ollama serve` when the router starts if Ollama is offline and its CLI is installed
- **Smart Auto-Routing** — Automatically classifies prompts and routes them to the best model using regex keyword heuristics
- **VRAM-Safe** — Enforces `num_ctx: 2048` on every Ollama request to prevent KV-cache overflow on low-VRAM GPUs (GTX 1650)
- **Think-Tag Processing** — Parses and optionally strips `<think>...</think>` blocks from DeepSeek reasoning output
- **Manual Override** — Bypass auto-routing and target any specific model directly
- **Health Monitoring** — Built-in health endpoint with Ollama connectivity check

---

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GPU with ≥ 4 GB VRAM (tested on GTX 1650) |
| **RAM** | 16 GB DDR4 recommended |
| **OS** | Windows 10/11 or Linux |
| **Ollama** | v0.20+ running locally on port `11434` |

---

## 📦 Installed Models & Roles

| Model | Size | Role |
|-------|------|------|
| `gemma3:1b` | 815 MB | Ultra-fast fallback, brief tasks |
| `qwen2.5:3b` | 1.9 GB | General conversation, summaries, long-form writing |
| `deepseek-r1:1.5b` | 1.1 GB | Reasoning, step-by-step logic, math |
| `qwen2.5-coder:1.5b-base` | 986 MB | Code autocompletion, script generation, programming |

### Install the models

```bash
ollama pull gemma3:1b
ollama pull qwen2.5:3b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:1.5b-base
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ardy644/ollama-model-router.git
cd ollama-model-router
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

This local service has no authentication. Keep it bound to `127.0.0.1`; do not expose port `8000` to your network or the public internet without adding access control.

If Ollama is not already running, the router starts `ollama serve` automatically when the CLI is available. If the CLI cannot be found, start Ollama yourself.

### 4. Test it

```bash
python test_client.py
```

---

## 📡 API Reference

### `POST /v1/chat`

Send a prompt and receive a routed model response.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | *(required)* | The user prompt to send |
| `history` | `array` | `[]` | Optional recent user/assistant messages to provide conversation context (up to 12 messages) |
| `model` | `string` | `"auto"` | Model to use: `auto` or one of the model choices in Settings |
| `strip_reasoning` | `bool` | `true` | If `true`, strips `<think>` tags from DeepSeek output and moves the trace to the `reasoning` field |
| `context_limit` | `int` | `2048` | Context window size (range: 512–4096). Controls VRAM usage via Ollama's `num_ctx` |

**Example — Auto-routing:**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort a list"}'
```

**Example — Manual model override:**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello", "model": "gemma3:1b"}'
```

**Response:**

```json
{
  "model": "qwen2.5-coder:1.5b-base",
  "response": "def sort_list(lst):\n    return sorted(lst)",
  "reasoning": null,
  "tokens": {
    "prompt_eval_count": 12,
    "eval_count": 45
  }
}
```

---

### `GET /v1/models`

Returns the configured model list and auto-routing rules. The `installed_models` field is a configured allowlist; it does not check which models are currently present in Ollama.

**Response:**

```json
{
  "installed_models": [
    "gemma3:1b",
    "qwen2.5:3b",
    "deepseek-r1:1.5b",
    "qwen2.5-coder:1.5b-base"
  ],
  "routing_options": ["auto", "gemma3:1b", "qwen2.5:3b", "deepseek-r1:1.5b", "qwen2.5-coder:1.5b-base"],
  "auto_routing_rules": {
    "coding_keywords": "python, code, function, def, class, sql, bug, ...",
    "reasoning_keywords": "solve, calculate, why, step-by-step, logic, ...",
    "default_fallback": "gemma3:1b"
  }
}
```

### `GET /v1/settings` and `PUT /v1/settings`

Read or update the model choices, coding/reasoning/fallback model mappings, and keyword lists used by automatic routing. Settings saved from the Web UI are stored in `router_settings.json` and loaded again when the router restarts. Model tags should match models available in your local Ollama installation.

---

### `GET /health`

Health check with Ollama connectivity status.

**Response:**

```json
{
  "status": "healthy",
  "ollama_connected": true
}
```

When Ollama cannot be reached, `/health` returns HTTP `503` with `"status": "unhealthy"` and `"ollama_connected": false`.

---

## 🧠 Auto-Routing Logic

When `model` is set to `"auto"` (the default), the router classifies the prompt using regex keyword matching:

```
Prompt
  ├─ Coding keywords matched?  →  qwen2.5-coder:1.5b-base
  │   (python, code, function, def, class, sql, bug, script,
  │    html, css, javascript, typescript, api, endpoint, etc.)
  │
  ├─ Reasoning keywords matched?  →  deepseek-r1:1.5b
  │   (solve, calculate, why, step-by-step, logic, evaluate,
  │    proof, theorem, math, equation, reason, derive, etc.)
  │
  └─ No match (fallback)  →  gemma3:1b
```

> **Priority:** Coding → Reasoning → General. If a prompt contains both coding and reasoning keywords, the coding model wins.

> `gemma3:1b` is the automatic fallback for prompts that do not match coding or reasoning keywords. You can also select it directly.

---

## 🔧 Error Handling

| Scenario | HTTP Status | Detail |
|----------|-------------|--------|
| Unknown/invalid model name | `400` | Model is not in the configured model choices |
| Ollama server offline | `503` | `"Ollama server is offline or unreachable"` |
| Ollama returns an error | `502` | Ollama's error status and message forwarded |
| `context_limit` out of range | `422` | Must be between 512 and 4096 |
| Unsupported message role | `422` | `/v1/chat/completions` accepts `system`, `user`, and `assistant` roles |

Temperature must be between `0` and `2`. On `/v1/chat/completions`, `max_tokens` sets Ollama's maximum generated-token count (`num_predict`) and must be between `1` and `4096`.

---

## 🧪 Test Client

`test_client.py` runs 6 happy-path checks and 4 validation checks against the running server:

| Check | What it covers |
|---|---|
| Happy path | Personas, editable routing settings, automatic code and reasoning routing, manual model selection, and the OpenAI-style endpoint |
| Validation | Empty prompts, unknown models, context limits, and an empty messages array |

```bash
python test_client.py
```

```
🚀 Ollama Model Router — Test Suite
   Target: http://127.0.0.1:8000
✅ Test 1 PASSED — routed to qwen2.5-coder:1.5b-base
✅ Test 2 PASSED — routed to deepseek-r1:1.5b
✅ Test 3 PASSED — routed to gemma3:1b
🎉 All 10 tests passed (6 Happy Path + 4 Negative/Edge Case)!
```

---

## 📁 Project Structure

```
ollama-model-router/
├── main.py              # FastAPI application, web UI, router, and API endpoints
├── test_client.py       # Standalone client for 10 happy-path and validation checks
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

---

## ⚙️ Configuration

Routing rules and model choices can be edited in the Web UI Settings panel. Other request defaults are:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Ollama URL | `http://127.0.0.1:11434` | Hardcoded in `main.py` (change `OLLAMA_BASE`) |
| FastAPI port | `8000` | Set via `uvicorn` CLI flag |
| Context limit | `2048` | VRAM-safe default; adjustable per-request (512–4096) |
| Ollama timeout | `120s` | Max wait for model inference |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
