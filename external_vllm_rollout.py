"""Helpers for routing GRPO rollouts to a stock vLLM server."""

from __future__ import annotations

import os
from typing import Any, Callable

import requests

RolloutFunc = Callable[[list[str], Any], dict[str, Any]]


def _maybe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


class NullVLLMClient:
    """No-op stand in so TRL skips vLLM weight syncing when using custom rollouts."""

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ) -> None:
        self.base_url = base_url or f"http://{host}:{server_port}"
        self.group_port = group_port
        self.connection_timeout = connection_timeout

    def init_communicator(self, **_: Any) -> None:  # noqa: ANN401 - signature parity
        return None

    def update_named_param(self, name: str, tensor) -> None:  # noqa: ANN001 - tensor is torch.Tensor
        return None

    def reset_prefix_cache(self) -> None:
        return None

    def close_communicator(self) -> None:
        return None

    def generate(self, *_, **__):  # pragma: no cover - should never be called
        raise RuntimeError("NullVLLMClient.generate should not be used when a rollout_func is provided")

    def chat(self, *_, **__):  # pragma: no cover - should never be called
        raise RuntimeError("NullVLLMClient.chat should not be used when a rollout_func is provided")


def make_openai_vllm_rollout(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 120.0,
    extra_payload: dict[str, Any] | None = None,
) -> RolloutFunc:
    """Return a rollout function that queries a regular ``vllm serve`` instance."""

    session = requests.Session()
    endpoint = base_url.rstrip("/") + "/v1/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _tokenize(tokenizer, text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        if isinstance(encoded, dict):
            return list(encoded.get("input_ids", []))
        return list(encoded)

    def rollout(prompts: list[str], trainer) -> dict[str, Any]:
        if not prompts:
            return {"prompt_ids": [], "completion_ids": [], "logprobs": []}

        tokenizer = getattr(trainer, "processing_class", None)
        if tokenizer is None:
            raise RuntimeError("Trainer does not expose a tokenizer for custom rollouts")

        mode = "train" if trainer.model.training else "eval"
        num_generations = trainer.num_generations if mode == "train" else (
            trainer.num_generations_eval or trainer.num_generations
        )

        prompt_ids: list[list[int]] = []
        for prompt in prompts:
            prompt_ids.append(_tokenize(tokenizer, prompt))

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompts,
            "n": num_generations,
            "max_tokens": trainer.max_completion_length,
            "temperature": trainer.temperature,
            "top_p": trainer.top_p,
            "top_k": trainer.top_k,
            "min_p": trainer.min_p,
            "repetition_penalty": trainer.repetition_penalty,
            "logprobs": 0,
            "stream": False,
            "return_token_ids": True,
            "echo": False,
        }
        if extra_payload:
            payload.update(extra_payload)
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Failed to reach vLLM server at {endpoint}: {exc}") from exc

        data = response.json()
        choices = data.get("choices", [])
        expected = len(prompts) * num_generations
        if len(choices) != expected:
            raise RuntimeError(
                f"vLLM server returned {len(choices)} choices, expected {expected}. "
                "Ensure --n matches num_generations and the server is not streaming."
            )

        completion_ids: list[list[int]] = []
        completion_logprobs: list[list[float]] = []

        for prompt_index in range(len(prompts)):
            offset = prompt_index * num_generations
            chunk = choices[offset : offset + num_generations]
            if chunk:
                prompt_token_ids = chunk[0].get("prompt_token_ids")
                if prompt_token_ids:
                    prompt_ids[prompt_index] = [int(tok) for tok in prompt_token_ids]
            for choice in chunk:
                token_ids = choice.get("token_ids")
                if token_ids is None:
                    token_ids = _tokenize(tokenizer, choice.get("text", ""))
                else:
                    token_ids = [int(tok) for tok in token_ids]
                completion_ids.append(token_ids)
                logprob_payload = ((choice.get("logprobs") or {}).get("token_logprobs"))
                if logprob_payload is None:
                    completion_logprobs.append([0.0] * len(token_ids))
                else:
                    completion_logprobs.append([_maybe_float(val) for val in logprob_payload])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": completion_logprobs,
        }

    return rollout


def resolve_vllm_base_url(config) -> str:
    if config.vllm_server_base_url:
        return config.vllm_server_base_url
    return f"http://{config.vllm_server_host}:{config.vllm_server_port}"


DEFAULT_VLLM_BASE = os.getenv("VLLM_API_BASE")
DEFAULT_VLLM_KEY = os.getenv("VLLM_API_KEY")
DEFAULT_VLLM_TIMEOUT = float(os.getenv("VLLM_API_TIMEOUT", "120"))
