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
import logging
import yaml
import threading
from typing import List, Protocol, runtime_checkable, Optional, Dict, Any

logger = logging.getLogger("knowledgeaudit.llm_client")

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str, num_ctx: Optional[int] = None) -> str: ...
    def batch_generate(self, prompts: List[str], num_ctx: Optional[int] = None) -> List[str]: ...


class OllamaClient:
    def __init__(self, model=None, base_url="http://localhost:11434"):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
        self.base_url = base_url
        self._local = threading.local()

    @property
    def last_usage(self):
        return getattr(self._local, "last_usage", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local.last_usage = value

    def generate(self, prompt: str, num_ctx: Optional[int] = None) -> str:
            endpoint = f"{self.base_url}/api/chat"
            
            # Context window precedence: 
            # 1. Per-call num_ctx
            # 2. Class default (8192)
            ctx_size = num_ctx if num_ctx is not None else 8192
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "keep_alive": -1,
                "options": {
                    "num_ctx": ctx_size,
                    "num_predict": 2048, 
                    "temperature": 0.0,
                    "num_gpu": 99,     
                }
            }
            
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=300,
                )
                
                # logger.debug("--- RAW LLM RESPONSE START ---")
                # logger.debug("Status: %s", response.status_code)
                # logger.debug("Body: %s", response.text)
                # logger.debug("--- RAW LLM RESPONSE END ---")

                if response.status_code == 404:
                    logger.critical("Ollama endpoint NOT FOUND at %s.", endpoint)
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                logger.critical("LLM Client Request Failed: %s", str(e))
                # If a 500 error occurs, log the partial response text if available
                if hasattr(e, 'response') and e.response is not None:
                    logger.error("Error Response Body: %s", e.response.text)
                raise

            data = response.json()
            
            # Monitor for silent truncation
            prompt_eval = data.get("prompt_eval_count", 0)
            if prompt_eval >= payload["options"]["num_ctx"]:
                logger.warning(
                    "CONTEXT OVERFLOW: prompt_eval_count (%d) >= num_ctx (%d). "
                    "The model's reasoning is likely truncated.",
                    prompt_eval, payload["options"]["num_ctx"]
                )

            message = data.get("message", {})
            content = message.get("content", "")
            
            self.last_usage = {
                "prompt_tokens": prompt_eval,
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": prompt_eval + data.get("eval_count", 0),
            }
            return content

    def batch_generate(self, prompts: List[str], num_ctx: Optional[int] = None) -> List[str]:
        """Sequential fallback for Ollama."""
        return [self.generate(p, num_ctx) for p in prompts]


class MLXClient:
    """Native MLX client for Mac M4 efficiency."""
    def __init__(self, model: str = None, num_ctx: int = 8192):
        self._local = threading.local()
        # Lazily import to avoid dependency on non-mac systems or when not using MLX
        try:
            from mlx_lm import load, generate, batch_generate
            self._load_fn = load
            self._generate_fn = generate
            self._batch_generate_fn = batch_generate
        except ImportError:
            logger.error("mlx-lm not installed. Run setup_mlx.sh")
            raise

        self.model_path = model or "mlx-community/Qwen2.5-14B-Instruct-4bit"
        self.default_num_ctx = num_ctx
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            logger.info("Loading MLX model: %s", self.model_path)
            self._model, self._tokenizer = self._load_fn(self.model_path)

    @property
    def last_usage(self):
        return getattr(self._local, "last_usage", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local.last_usage = value

    def generate(self, prompt: str, num_ctx: Optional[int] = None) -> str:
        self._ensure_model()
        
        # Parity with OllamaClient context handling
        ctx_size = num_ctx if num_ctx is not None else self.default_num_ctx
        
        # MLX generate options
        # Note: mlx_lm.generate usually takes max_tokens, temp, etc.
        # We simulate the chat/json structure if possible, 
        # but for now we'll match the prompt expectation.
        
        messages = [{"role": "user", "content": prompt}]
        # Helper to format for Qwen/Instruct models if needed, 
        # but mlx_lm handle chat tempaltes if tokenizer has one.
        
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Token counting for prompt
        prompt_tokens = len(self._tokenizer.encode(formatted_prompt))
        
        if prompt_tokens >= ctx_size:
            logger.warning("CONTEXT OVERFLOW: prompt tokens (%d) >= num_ctx (%d)", 
                           prompt_tokens, ctx_size)

        response_text = self._generate_fn(
            self._model, 
            self._tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=2048,
            verbose=False
        )

        completion_tokens = len(self._tokenizer.encode(response_text))
        
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        
        return response_text

    def batch_generate(self, prompts: List[str], num_ctx: Optional[int] = None) -> List[str]:
        """Sequential fallback for MLX to avoid batch processing issues."""
        return [self.generate(p, num_ctx) for p in prompts]


class MockLLMClient:
    """Generic mock -- routes by substring matching on prompt."""
    def __init__(self, responses: dict = None):
        self.responses = responses or {}
        self.call_count = 0
        self._local = threading.local()

    @property
    def last_usage(self):
        return getattr(self._local, "last_usage", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local.last_usage = value

    def generate(self, prompt: str, num_ctx: Optional[int] = None) -> str:
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

    def batch_generate(self, prompts: List[str], num_ctx: Optional[int] = None) -> List[str]:
        """Sequential fallback for Mock."""
        return [self.generate(p, num_ctx) for p in prompts]


def _load_config() -> Dict[str, Any]:
    config_path = os.path.join(_PROJECT_ROOT, "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Error loading config.yaml: %s", e)
    return {}


def get_client(provider: str = None) -> LLMClient:
    config = _load_config()
    llm_cfg = config.get("llm", {})
    
    # Precedence: 1. Argument, 2. Env Var, 3. Config File, 4. Default
    provider = provider or os.getenv("LLM_PROVIDER") or llm_cfg.get("provider", "mock")
    
    logger.info(f"debug_token: Initializing LLM Client with provider: {provider}")
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL") or llm_cfg.get("model") or "qwen2.5:14b"
        return OllamaClient(model=model)
    elif provider == "mlx":
        mlx_cfg = config.get("mlx", {})
        model = mlx_cfg.get("model")
        num_ctx = mlx_cfg.get("num_ctx", 8192)
        return MLXClient(model=model, num_ctx=num_ctx)
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


class SentenceTransformerClient:
    """Local embedding client using SentenceTransformers."""
    def __init__(self, model: str = None):
        self._local = threading.local()
        try:
            from sentence_transformers import SentenceTransformer
            self._loader = SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Run 'pip install sentence-transformers'")
            raise

        self.model_path = model or "nomic-ai/nomic-embed-text-v1.5"
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            logger.info("Loading Embedding model: %s", self.model_path)
            # trust_remote_code is often needed for Nomic/newer models
            self._model = self._loader(self.model_path, trust_remote_code=True)

    @property
    def last_usage(self):
        return getattr(self._local, "last_usage", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local.last_usage = value

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        # Nomic v1.5 technically prefers prefixes 'search_query: ' / 'search_document: '
        # But without a clean way to distinguish in this protocol, we use raw text.
        # This works but may be slightly suboptimal compared to prefixed.
        
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Convert numpy to list of lists
        results = vectors.tolist()
        
        # Token counting (approximate / using internal tokenizer)
        total_tokens = 0
        if hasattr(self._model, "tokenizer"):
            for t in texts:
                total_tokens += len(self._model.tokenizer.encode(t))
        
        self.last_usage = {
            "prompt_tokens": total_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens,
        }
        return results


class MockEmbeddingClient:
    """Deterministic mock embedding client for testing.

    Produces reproducible vectors by hashing input text, so the same
    text always yields the same embedding.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0
        self._local = threading.local()

    @property
    def last_usage(self):
        return getattr(self._local, "last_usage", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local.last_usage = value

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
        self._local = threading.local()

    @property
    def last_usage(self):
        return getattr(self._local, "last_usage", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local.last_usage = value

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
    config = _load_config()
    embed_cfg = config.get("embedding", {})
    
    provider = provider or os.getenv("EMBEDDING_PROVIDER") or embed_cfg.get("provider", "mock")
    
    if provider == "ollama":
        model = os.getenv("OLLAMA_EMBED_MODEL") or embed_cfg.get("model") or "nomic-embed-text"
        return OllamaEmbeddingClient(model=model)
    elif provider == "sentence_transformers":
        st_cfg = config.get("sentence_transformers", {})
        model = st_cfg.get("model") or embed_cfg.get("model")
        return SentenceTransformerClient(model=model)
    elif provider == "mock":
        return MockEmbeddingClient()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
