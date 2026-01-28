"""
LLM Client abstraction layer.
Location: llm_client.py

Provides OllamaClient for local LLM inference, a generic MockLLMClient,
and a get_client() factory function.

v5 additions: EmbeddingClient protocol, MockEmbeddingClient,
OllamaEmbeddingClient, and get_embedding_client() factory.
"""

import os
import hashlib
import struct
import requests
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class OllamaClient:
    def __init__(self, model=None, base_url="http://localhost:11434"):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        self.base_url = base_url
        self.last_usage = None

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        self.last_usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0)
                         + data.get("eval_count", 0),
        }
        return data["response"]


class MockLLMClient:
    """Generic mock -- routes by substring matching on prompt."""
    def __init__(self, responses: dict = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_usage = None

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        for key, response in self.responses.items():
            if key in prompt:
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(response) // 4
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                return response
        result = '{"error": "no mock response configured"}'
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(result) // 4
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return result


def get_client(provider: str = None) -> LLMClient:
    provider = provider or os.getenv("LLM_PROVIDER", "mock")
    if provider == "ollama":
        return OllamaClient()
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# EMBEDDING CLIENT (v5)
# ============================================================================

@runtime_checkable
class EmbeddingClient(Protocol):
    """Embedding client interface for vector operations (v5)."""
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class MockEmbeddingClient:
    """Deterministic mock embedding client for testing.

    Produces reproducible vectors by hashing input text, so the same
    text always yields the same embedding.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0
        self.last_usage = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        total_prompt_tokens = sum(len(t) // 4 for t in texts)
        self.last_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_prompt_tokens,
        }
        return [self._deterministic_vector(t) for t in texts]

    def _deterministic_vector(self, text: str) -> List[float]:
        """Hash text into a deterministic float vector of self.dimension."""
        digest = hashlib.sha256(text.encode()).digest()
        # Expand digest to fill dimension by repeating hash with counters
        floats: List[float] = []
        for i in range(self.dimension):
            block = hashlib.md5(digest + struct.pack(">I", i)).digest()
            # Take first 4 bytes as a float in [-1, 1]
            raw = struct.unpack(">I", block[:4])[0]
            floats.append((raw / 0xFFFFFFFF) * 2 - 1)
        # Normalize to unit length
        magnitude = sum(f * f for f in floats) ** 0.5
        if magnitude > 0:
            floats = [f / magnitude for f in floats]
        return floats


class OllamaEmbeddingClient:
    """Embedding client using Ollama's /api/embeddings endpoint."""

    def __init__(self, model: str = None, base_url: str = "http://localhost:11434"):
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.base_url = base_url
        self.last_usage = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        total_prompt_tokens = 0
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            results.append(data["embedding"])
            total_prompt_tokens += data.get("prompt_eval_count", 0)
        self.last_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_prompt_tokens,
        }
        return results


def get_embedding_client(provider: str = None) -> EmbeddingClient:
    """Factory for embedding clients."""
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "mock")
    if provider == "ollama":
        return OllamaEmbeddingClient()
    elif provider == "mock":
        return MockEmbeddingClient()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
