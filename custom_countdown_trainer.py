"""
Custom Countdown RL trainer without TRL's GRPOTrainer.

This keeps the same overall idea:
- sample multiple completions per prompt
- score completions with reward functions
- normalize rewards within each prompt group (GRPO-style advantages)
- apply a CISPO-style policy update in plain torch
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown import (
    eval_arithmetic,
    eval_expression,
    generate_puzzles,
    is_give_up,
    parse_solution,
)


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "countdown-custom"
    seed: int = 42
    train_examples: int = 500
    eval_examples: int = 100
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 2
    max_completion_length: int = 128
    learning_rate: float = 1e-6
    temperature: float = 1.0
    reward_exact: float = 1.0
    reward_close: float = 0.3
    reward_format: float = 0.1
    cispo_epsilon_low: float = 1e6
    cispo_epsilon_high: float = 8.0
    max_grad_norm: float = 1.0
    eval_every: int = 25
    save_every: int = 25
    dtype: str = "bf16"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    # default bf16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def normalize_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std + 1e-8) for r in rewards]


def compute_reward_components(text: str, numbers: list[int], target: int) -> tuple[float, float, float]:
    sol = parse_solution(text)

    # Exact
    exact = 0.0
    if sol and not is_give_up(sol):
        result = eval_expression(sol, numbers)
        if result is not None and abs(result - target) < 1e-6:
            exact = 1.0

    # Closeness
    close = 0.0
    if sol and not is_give_up(sol):
        result = eval_arithmetic(sol)
        if result is not None:
            close = 0.5 ** (abs(result - target) / 10)

    # Format
    fmt = 0.0
    if "<reasoning>" in text and "</reasoning>" in text:
        fmt += 0.5
    if sol and not is_give_up(sol) and any(op in sol for op in ("+", "-", "*", "/")):
        fmt += 0.5

    return exact, close, fmt


def weighted_reward(
    components: tuple[float, float, float],
    weights: tuple[float, float, float],
) -> float:
    return (
        components[0] * weights[0]
        + components[1] * weights[1]
        + components[2] * weights[2]
    )


def strip_trailing_pad(tokens: list[int], pad_id: int | None) -> list[int]:
    if pad_id is None:
        return tokens
    end = len(tokens)
    while end > 0 and tokens[end - 1] == pad_id:
        end -= 1
    return tokens[:end]


@torch.no_grad()
def rollout_group(
    model,
    tokenizer,
    prompt_messages: list[dict],
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    device: torch.device,
) -> tuple[list[int], list[list[int]], list[str]]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_mask = torch.ones_like(prompt_tensor)

    generated = model.generate(
        input_ids=prompt_tensor,
        attention_mask=prompt_mask,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        num_return_sequences=num_generations,
        max_new_tokens=max_completion_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = prompt_tensor.shape[1]
    completion_token_lists: list[list[int]] = []
    completion_texts: list[str] = []
    for row in generated:
        completion_ids = row[prompt_len:].tolist()
        completion_ids = strip_trailing_pad(completion_ids, tokenizer.pad_token_id)
        completion_token_lists.append(completion_ids)
        completion_texts.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

    return prompt_ids, completion_token_lists, completion_texts


def build_rl_batch(
    prompt_ids_batch: list[list[int]],
    completion_ids_batch: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_sequences: list[list[int]] = []
    for prompt_ids, completion_ids in zip(prompt_ids_batch, completion_ids_batch):
        full_sequences.append(prompt_ids + completion_ids)

    max_len = max(len(seq) for seq in full_sequences)
    batch_size = len(full_sequences)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    completion_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.float32, device=device)

    for i, (prompt_ids, completion_ids, seq) in enumerate(
        zip(prompt_ids_batch, completion_ids_batch, full_sequences)
    ):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :seq_len] = 1

        if completion_ids:
            start = len(prompt_ids) - 1
            end = start + len(completion_ids)
            completion_mask[i, start:end] = 1.0

    return input_ids, attention_mask, completion_mask


def compute_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = shift_logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs


def cispo_loss(
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = new_token_logprobs - old_token_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high).detach()

    adv = advantages.unsqueeze(-1)
    weighted_logp = clipped_ratio * adv * new_token_logprobs * completion_mask
    token_counts = completion_mask.sum(dim=-1).clamp(min=1.0)
    per_sample_obj = weighted_logp.sum(dim=-1) / token_counts
    loss = -per_sample_obj.mean()

    with torch.no_grad():
        active = completion_mask > 0
        if active.any():
            ratio_vals = ratio[active]
            log_ratio_vals = log_ratio[active]
            clip_frac = (
                (ratio_vals > (1.0 + epsilon_high)) | (ratio_vals < (1.0 - epsilon_low))
            ).float().mean()
            mismatch_kl = (ratio_vals - log_ratio_vals - 1.0).mean()
            ratio_mean = ratio_vals.mean()
        else:
            clip_frac = torch.zeros((), device=loss.device)
            mismatch_kl = torch.zeros((), device=loss.device)
            ratio_mean = torch.zeros((), device=loss.device)

    stats = {
        "loss": float(loss.detach().cpu()),
        "ratio_mean": float(ratio_mean.detach().cpu()),
        "clip_frac": float(clip_frac.detach().cpu()),
        "kl_actor_policy": float(mismatch_kl.detach().cpu()),
    }
    return loss, stats


@torch.no_grad()
def evaluate_exact_match(
    model,
    tokenizer,
    eval_dataset,
    max_eval_examples: int,
    max_completion_length: int,
    device: torch.device,
) -> float:
    n = min(len(eval_dataset), max_eval_examples)
    if n == 0:
        return 0.0

    hits = 0
    for i in range(n):
        row = eval_dataset[i]
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        prompt_mask = torch.ones_like(prompt_tensor)

        generated = model.generate(
            input_ids=prompt_tensor,
            attention_mask=prompt_mask,
            do_sample=False,
            max_new_tokens=max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids = generated[0, prompt_tensor.shape[1] :].tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        exact, _, _ = compute_reward_components(
            completion_text, row["numbers"], row["target"]
        )
        if exact > 0.5:
            hits += 1
    return hits / n


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(cfg.dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    train_dataset = generate_puzzles(cfg.train_examples, seed=cfg.seed)
    eval_dataset = generate_puzzles(cfg.eval_examples, seed=cfg.seed + 81)

    os.makedirs(cfg.output_dir, exist_ok=True)

    reward_weights = (cfg.reward_exact, cfg.reward_close, cfg.reward_format)
    rng = random.Random(cfg.seed + 999)

    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_reward = 0.0
        step_exact = 0.0
        rollout_count = 0

        for _ in range(cfg.gradient_accumulation_steps):
            prompt_batch: list[list[int]] = []
            completion_batch: list[list[int]] = []
            advantage_batch: list[float] = []

            micro_rewards: list[float] = []
            micro_exacts: list[float] = []

            model.eval()
            for _ in range(cfg.per_device_train_batch_size):
                row = train_dataset[rng.randrange(len(train_dataset))]
                prompt_ids, completion_ids_group, completion_texts = rollout_group(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_messages=row["prompt"],
                    num_generations=cfg.num_generations,
                    max_completion_length=cfg.max_completion_length,
                    temperature=cfg.temperature,
                    device=device,
                )

                group_rewards: list[float] = []
                group_exacts: list[float] = []
                for completion_ids, completion_text in zip(completion_ids_group, completion_texts):
                    exact, close, fmt = compute_reward_components(
                        completion_text, row["numbers"], row["target"]
                    )
                    score = weighted_reward((exact, close, fmt), reward_weights)
                    group_rewards.append(score)
                    group_exacts.append(exact)

                    # Skip empty completions (no tokens to train on).
                    if completion_ids:
                        prompt_batch.append(prompt_ids)
                        completion_batch.append(completion_ids)

                group_advantages = normalize_advantages(group_rewards)

                for completion_ids, adv in zip(completion_ids_group, group_advantages):
                    if completion_ids:
                        advantage_batch.append(adv)

                micro_rewards.extend(group_rewards)
                micro_exacts.extend(group_exacts)

            if not prompt_batch:
                continue

            input_ids, attention_mask, completion_mask = build_rl_batch(
                prompt_ids_batch=prompt_batch,
                completion_ids_batch=completion_batch,
                pad_id=tokenizer.pad_token_id,
                device=device,
            )

            with torch.no_grad():
                old_token_logprobs = compute_token_logprobs(model, input_ids, attention_mask).detach()

            model.train()
            new_token_logprobs = compute_token_logprobs(model, input_ids, attention_mask)
            advantages = torch.tensor(
                advantage_batch,
                dtype=new_token_logprobs.dtype,
                device=device,
            )

            loss, _ = cispo_loss(
                new_token_logprobs=new_token_logprobs,
                old_token_logprobs=old_token_logprobs,
                completion_mask=completion_mask,
                advantages=advantages,
                epsilon_low=cfg.cispo_epsilon_low,
                epsilon_high=cfg.cispo_epsilon_high,
            )

            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach().cpu())
            step_reward += sum(micro_rewards)
            step_exact += sum(micro_exacts)
            rollout_count += len(micro_rewards)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        avg_reward = (step_reward / rollout_count) if rollout_count else 0.0
        avg_exact = (step_exact / rollout_count) if rollout_count else 0.0
        print(
            f"step={step:04d} loss={step_loss:.4f} avg_reward={avg_reward:.4f} "
            f"exact_rate={avg_exact:.4f} grad_norm={float(grad_norm):.4f} rollouts={rollout_count}"
        )

        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            model.eval()
            eval_exact = evaluate_exact_match(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                max_eval_examples=min(50, len(eval_dataset)),
                max_completion_length=cfg.max_completion_length,
                device=device,
            )
            print(f"[eval] step={step:04d} exact_match={eval_exact:.4f}")
            model.train()

        if cfg.save_every > 0 and step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved_final={final_dir}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Custom Countdown RL trainer")
    parser.add_argument("--model-name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--train-examples", type=int, default=TrainConfig.train_examples)
    parser.add_argument("--eval-examples", type=int, default=TrainConfig.eval_examples)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=TrainConfig.per_device_train_batch_size,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--num-generations", type=int, default=TrainConfig.num_generations)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=TrainConfig.max_completion_length,
    )
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--reward-exact", type=float, default=TrainConfig.reward_exact)
    parser.add_argument("--reward-close", type=float, default=TrainConfig.reward_close)
    parser.add_argument("--reward-format", type=float, default=TrainConfig.reward_format)
    parser.add_argument("--cispo-epsilon-low", type=float, default=TrainConfig.cispo_epsilon_low)
    parser.add_argument("--cispo-epsilon-high", type=float, default=TrainConfig.cispo_epsilon_high)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default=TrainConfig.dtype,
    )

    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        reward_exact=args.reward_exact,
        reward_close=args.reward_close,
        reward_format=args.reward_format,
        cispo_epsilon_low=args.cispo_epsilon_low,
        cispo_epsilon_high=args.cispo_epsilon_high,
        max_grad_norm=args.max_grad_norm,
        eval_every=args.eval_every,
        save_every=args.save_every,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    train(parse_args())
