# 🚀 Ollama Model Router

A lightweight, production-ready **FastAPI** orchestration layer that sits in front of a local [Ollama](https://ollama.com) server. It intelligently auto-routes prompts to the best local LLM based on keyword heuristics, enforces VRAM-safe context limits, and strips DeepSeek `<think>` reasoning tags.

Built for **constrained hardware** — designed to run comfortably on a 4 GB VRAM GPU.

---

## ✨ Features

- **Smart Auto-Routing** — Automatically classifies prompts and routes them to the best model using regex keyword heuristics
- **VRAM-Safe** — Enforces `num_ctx` on every Ollama request to prevent KV-cache overflow on low-VRAM GPUs
- **Think-Tag Processing** — Parses and optionally strips `<think>...</think>` blocks from DeepSeek reasoning output
- **Manual Override** — Bypass auto-routing and target any specific model directly
- **Health Monitoring** — Built-in health endpoint with Ollama connectivity check
- **Async** — Fully asynchronous request forwarding via `httpx.AsyncClient`

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
git clone https://github.com/<your-username>/ollama-model-router.git
cd ollama-model-router
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure Ollama is running

```bash
ollama serve
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Test it

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
| `model` | `string` | `"auto"` | Model to use. One of: `auto`, `gemma3:1b`, `qwen2.5:3b`, `deepseek-r1:1.5b`, `qwen2.5-coder:1.5b-base` |
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

Returns the list of installed models and auto-routing rules.

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
    "default_fallback": "qwen2.5:3b"
  }
}
```

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
  └─ No match (fallback)  →  qwen2.5:3b
```

> **Priority:** Coding → Reasoning → General. If a prompt contains both coding and reasoning keywords, the coding model wins.

> **Note:** `gemma3:1b` is never auto-selected — it's available only via manual override (`"model": "gemma3:1b"`).

---

## 🔧 Error Handling

| Scenario | HTTP Status | Detail |
|----------|-------------|--------|
| Unknown/invalid model name | `422` | Pydantic validation error |
| Ollama server offline | `503` | `"Ollama server is offline or unreachable"` |
| Ollama returns an error | `502` | Ollama's error status and message forwarded |
| `context_limit` out of range | `422` | Must be between 512 and 4096 |

---

## 🧪 Test Client

`test_client.py` runs 3 end-to-end tests against the running server:

| # | Test | Expected Model |
|---|------|---------------|
| 1 | Auto code routing (`"Write a Python function..."`) | `qwen2.5-coder:1.5b-base` |
| 2 | Auto reasoning routing (`"Solve step by step..."`) | `deepseek-r1:1.5b` |
| 3 | Manual override | `gemma3:1b` |

```bash
python test_client.py
```

```
🚀 Ollama Model Router — Test Suite
   Target: http://127.0.0.1:8000
✅ Test 1 PASSED — routed to qwen2.5-coder:1.5b-base
✅ Test 2 PASSED — routed to deepseek-r1:1.5b
✅ Test 3 PASSED — routed to gemma3:1b
🎉 All 3 tests passed!
```

---

## 📁 Project Structure

```
ollama-model-router/
├── main.py              # FastAPI application (3 endpoints, router, think-tag processor)
├── test_client.py       # Standalone test client (3 test scenarios)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Configuration

All configuration is done via request parameters — no config files needed. Key defaults:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Ollama URL | `http://127.0.0.1:11434` | Hardcoded in `main.py` (change `OLLAMA_BASE`) |
| FastAPI port | `8000` | Set via `uvicorn` CLI flag |
| Context limit | `2048` | VRAM-safe default; adjustable per-request (512–4096) |
| Ollama timeout | `120s` | Max wait for model inference |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
