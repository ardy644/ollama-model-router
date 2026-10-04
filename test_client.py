"""
Test Client for Ollama Model Router
-------------------------------------
Standalone script that exercises the three main routing scenarios:
  1. Auto-routing a coding prompt   -> qwen2.5-coder:1.5b-base
  2. Auto-routing a reasoning prompt -> deepseek-r1:1.5b  (with <think> extraction)
  3. Manual model override           -> gemma3:1b
"""

import sys
import io

# Fix Windows console encoding so emoji/unicode prints don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120.0  # generous timeout for slow local models


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# --------------------------------------------------------------------------
# Test 1: Auto-routing — Coding prompt
# --------------------------------------------------------------------------

def test_auto_code_routing():
    separator("Test 1: Auto Code Routing")
    payload = {
        "prompt": "Write a Python function to reverse a string",
    }
    r = httpx.post(f"{BASE_URL}/v1/chat", json=payload, timeout=TIMEOUT)
    data = r.json()

    print(f"  Status:  {r.status_code}")
    print(f"  Model:   {data.get('model')}")
    print(f"  Response: {data.get('response', '')[:200]}...")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert data["model"] == "qwen2.5-coder:1.5b-base", (
        f"Expected qwen2.5-coder:1.5b-base, got {data['model']}"
    )
    print("  ✅ PASSED — routed to qwen2.5-coder:1.5b-base")


# --------------------------------------------------------------------------
# Test 2: Auto-routing — Reasoning prompt with <think> extraction
# --------------------------------------------------------------------------

def test_auto_reasoning_routing():
    separator("Test 2: Auto Reasoning Routing + Think-Tag Extraction")
    payload = {
        "prompt": "Solve step by step: what is 23 * 47?",
        "strip_reasoning": True,
    }
    r = httpx.post(f"{BASE_URL}/v1/chat", json=payload, timeout=TIMEOUT)
    data = r.json()

    print(f"  Status:    {r.status_code}")
    print(f"  Model:     {data.get('model')}")
    print(f"  Response:  {data.get('response', '')[:200]}...")
    print(f"  Reasoning: {('present' if data.get('reasoning') else 'absent')}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert data["model"] == "deepseek-r1:1.5b", (
        f"Expected deepseek-r1:1.5b, got {data['model']}"
    )
    print("  ✅ PASSED — routed to deepseek-r1:1.5b")


# --------------------------------------------------------------------------
# Test 3: Manual model override — gemma3:1b
# --------------------------------------------------------------------------

def test_manual_model_override():
    separator("Test 3: Manual Model Override → gemma3:1b")
    payload = {
        "prompt": "Say hello in one sentence.",
        "model": "gemma3:1b",
    }
    r = httpx.post(f"{BASE_URL}/v1/chat", json=payload, timeout=TIMEOUT)
    data = r.json()

    print(f"  Status:  {r.status_code}")
    print(f"  Model:   {data.get('model')}")
    print(f"  Response: {data.get('response', '')[:200]}...")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert data["model"] == "gemma3:1b", (
        f"Expected gemma3:1b, got {data['model']}"
    )
    print("  ✅ PASSED — routed to gemma3:1b")


def test_personas_endpoint():
    separator("Test 4: Model Customization & Personas List")
    r = httpx.get(f"{BASE_URL}/v1/personas", timeout=10.0)
    data = r.json()

    print(f"  Status:   {r.status_code}")
    print(f"  Personas: {list(data.get('personas', {}).keys())}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert "coder" in data["personas"], "Expected 'coder' persona"
    assert "reasoner" in data["personas"], "Expected 'reasoner' persona"
    print("  ✅ PASSED — personas configured and accessible")


def test_openai_chat_completions():
    separator("Test 5: OpenAI-Compatible /v1/chat/completions")
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
        ],
        "temperature": 0.1,
    }
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = r.json()

    print(f"  Status:  {r.status_code}")
    print(f"  ID:      {data.get('id')}")
    print(f"  Object:  {data.get('object')}")
    print(f"  Model:   {data.get('model')}")
    print(f"  Reply:   {data.get('choices', [{}])[0].get('message', {}).get('content')}")

    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert data["object"] == "chat.completion", f"Expected chat.completion, got {data.get('object')}"
    assert len(data.get("choices", [])) > 0, "Expected at least one choice"
    print("  ✅ PASSED — OpenAI compatibility verified")


# --------------------------------------------------------------------------
# Negative & Resilience Tests
# --------------------------------------------------------------------------

def test_rejection_empty_or_whitespace_prompt():
    separator("Negative Test 1: Rejection of Empty/Whitespace Prompt")

    # 1. Empty string prompt
    payload_empty = {"prompt": ""}
    r_empty = httpx.post(f"{BASE_URL}/v1/chat", json=payload_empty, timeout=10.0)
    print(f"  Empty prompt status:      {r_empty.status_code}")
    assert r_empty.status_code == 422, f"Expected 422 for empty prompt, got {r_empty.status_code}"

    # 2. Whitespace-only prompt
    payload_whitespace = {"prompt": "   \n\t   "}
    r_ws = httpx.post(f"{BASE_URL}/v1/chat", json=payload_whitespace, timeout=10.0)
    print(f"  Whitespace prompt status: {r_ws.status_code}")
    assert r_ws.status_code == 422, f"Expected 422 for whitespace prompt, got {r_ws.status_code}"

    # 3. Missing prompt field
    r_missing = httpx.post(f"{BASE_URL}/v1/chat", json={}, timeout=10.0)
    print(f"  Missing prompt status:    {r_missing.status_code}")
    assert r_missing.status_code == 422, f"Expected 422 for missing prompt, got {r_missing.status_code}"

    print("  ✅ PASSED — empty/whitespace/missing prompt correctly rejected with 422")


def test_rejection_invalid_model():
    separator("Negative Test 2: Rejection of Invalid Model Name")
    payload = {
        "prompt": "Hello world",
        "model": "non-existent-gpt-99",
    }
    r = httpx.post(f"{BASE_URL}/v1/chat", json=payload, timeout=10.0)
    print(f"  Invalid model status: {r.status_code}")
    print(f"  Detail: {r.json().get('detail')}")

    assert r.status_code in (400, 422), f"Expected 400 or 422, got {r.status_code}"
    print(f"  ✅ PASSED — invalid model correctly rejected with {r.status_code}")


def test_out_of_bounds_context_limit():
    separator("Negative Test 3: Out-of-Bounds Context Limit Bounds (512 - 4096)")

    # 1. Lower bound violation (context_limit < 512)
    payload_low = {
        "prompt": "Hello world",
        "context_limit": 100,
    }
    r_low = httpx.post(f"{BASE_URL}/v1/chat", json=payload_low, timeout=10.0)
    print(f"  context_limit=100 status: {r_low.status_code}")
    assert r_low.status_code == 422, f"Expected 422 for context_limit=100, got {r_low.status_code}"

    # 2. Upper bound violation (context_limit > 4096)
    payload_high = {
        "prompt": "Hello world",
        "context_limit": 10000,
    }
    r_high = httpx.post(f"{BASE_URL}/v1/chat", json=payload_high, timeout=10.0)
    print(f"  context_limit=10000 status: {r_high.status_code}")
    assert r_high.status_code == 422, f"Expected 422 for context_limit=10000, got {r_high.status_code}"

    # 3. Completions endpoint lower bound violation
    payload_comp = {
        "messages": [{"role": "user", "content": "hi"}],
        "context_limit": 100,
    }
    r_comp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload_comp, timeout=10.0)
    print(f"  /v1/chat/completions context_limit=100 status: {r_comp.status_code}")
    assert r_comp.status_code == 422, f"Expected 422 for completions context_limit=100, got {r_comp.status_code}"

    print("  ✅ PASSED — out-of-bounds context limits correctly rejected with 422")


def test_rejection_empty_messages_array():
    separator("Negative Test 4: Rejection of Empty Messages in /v1/chat/completions")
    payload = {
        "model": "auto",
        "messages": [],
    }
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=10.0)
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Detail: {data.get('detail')}")

    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    assert data.get("detail") == "Messages array cannot be empty", (
        f"Unexpected detail message: {data.get('detail')}"
    )
    print("  ✅ PASSED — empty messages array correctly rejected with 400")


# --------------------------------------------------------------------------
# Run all tests
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Ollama Model Router — Test Suite")
    print(f"   Target: {BASE_URL}")

    try:
        # Happy Path Tests
        test_personas_endpoint()
        test_auto_code_routing()
        test_auto_reasoning_routing()
        test_manual_model_override()
        test_openai_chat_completions()

        # Negative & Resilience Tests
        test_rejection_empty_or_whitespace_prompt()
        test_rejection_invalid_model()
        test_out_of_bounds_context_limit()
        test_rejection_empty_messages_array()
    except httpx.ConnectError:
        print("\n❌ FAILED — Could not connect to the FastAPI server.")
        print(f"   Make sure it's running at {BASE_URL}")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n❌ FAILED — {e}")
        sys.exit(1)

    print("\n🎉 All 9 tests passed (5 Happy Path + 4 Negative/Edge Case)!")
