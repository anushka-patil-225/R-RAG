"""
LLM interface module.
Supports Groq (online) and Ollama (offline) with automatic fallback.
API key is read from environment or set directly.
"""

import os
import time
import requests

# --- Configuration ---
MODE = "online"  # "online" uses Groq, "offline" uses Ollama

OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "phi3:mini"

# Set your Groq API key here OR via environment variable GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama3-8b-8192"


def call_groq(prompt, system_prompt=None, max_tokens=1024):
    """Call Groq cloud API with retry logic (up to 2 attempts)."""
    if not GROQ_API_KEY:
        return None

    for attempt in range(2):
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,   # lower temperature = more factual, less creative
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.strip()

            if not content:
                print(f"[GROQ WARN] Empty response on attempt {attempt + 1}, retrying...")
                time.sleep(2)
                continue

            return content

        except Exception as e:
            print(f"[GROQ ERROR] Attempt {attempt + 1}: {e}")
            if attempt < 1:
                time.sleep(2)

    return None


def call_ollama(prompt, system_prompt=None, max_tokens=800):
    """Call local Ollama API."""
    try:
        payload = {
            "model": LOCAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # lower temperature = more factual
                "num_predict": max_tokens,
                "top_p": 0.9
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        res = requests.post(OLLAMA_URL, json=payload, timeout=180)
        data = res.json()
        content = data.get("response", "").strip()
        return content if content else None
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return None


def call_llm(prompt, system_prompt=None, max_tokens=1024):
    """
    Call the configured LLM with fallback.
    Returns response string or error message.
    """
    if MODE == "online":
        result = call_groq(prompt, system_prompt, max_tokens)
        if result:
            return result
        print("[WARN] Groq failed, falling back to Ollama...")
        result = call_ollama(prompt, system_prompt, max_tokens)
        if result:
            return result
    else:
        result = call_ollama(prompt, system_prompt, max_tokens)
        if result:
            return result
        print("[WARN] Ollama failed, falling back to Groq...")
        result = call_groq(prompt, system_prompt, max_tokens)
        if result:
            return result

    return "❌ LLM unavailable. Check your API key or Ollama connection."
