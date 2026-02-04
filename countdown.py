"""
Countdown Numbers Game - GRPO Training with TRL

Trains a language model to solve arithmetic countdown puzzles using
Group Relative Policy Optimization (GRPO).

Reproduction of: https://samikhan.ai/blog/countdown-rl.html
"""

import random
import re

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You solve countdown number puzzles. Given a list of numbers and a target, "
    "find an arithmetic expression using the given numbers that equals the target. "
    "Each number may only be used once. Use +, -, *, / operations.\n\n"
    "Respond with your reasoning in <reasoning>...</reasoning> tags, "
    "then your final equation in <solution>...</solution> tags."
)

SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL)
GIVE_UP_PHRASES = ["unable", "cannot", "impossible", "no solution", "not possible"]
LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = list(range(1, 11))


# ── Dataset ────────────────────────────────────────────────────────


def sample_numbers(rng: random.Random) -> list[int]:
    """Sample numbers like the real Countdown game: 1-3 large + rest small."""
    num_large = rng.randint(1, 3)
    large = rng.sample(LARGE_NUMBERS, num_large)
    small = rng.choices(SMALL_NUMBERS, k=6 - num_large)
    numbers = large + small
    rng.shuffle(numbers)
    return numbers


def generate_puzzles(n: int, seed: int = 42) -> Dataset:
    """Generate n solvable countdown puzzles by constructing solutions first."""
    rng = random.Random(seed)
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b if b != 0 and a % b == 0 else None),
    ]
    rows = []

    while len(rows) < n:
        numbers = sample_numbers(rng)
        pool = list(numbers)
        rng.shuffle(pool)

        k = rng.randint(2, min(4, len(pool)))
        result = pool[0]
        for i in range(1, k):
            _, fn = rng.choice(ops)
            new_result = fn(result, pool[i])
            if new_result is None:
                break
            result = int(new_result)
        else:
            if not (10 <= result <= 999):
                continue

            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Numbers: {numbers}\nTarget: {result}",
                        },
                    ],
                    "target": result,
                    "numbers": numbers,
                }
            )

    return Dataset.from_list(rows)


# ── Reward helpers ─────────────────────────────────────────────────


def parse_solution(text: str) -> str | None:
    m = SOLUTION_RE.search(text)
    return m.group(1).strip() if m else None


def eval_expression(expr: str, allowed: list[int]) -> float | None:
    """Evaluate a simple arithmetic expression, checking only allowed numbers are used."""
    if not re.fullmatch(r"[\d+\-*/().\s]+", expr):
        return None
    used = [int(n) for n in re.findall(r"\d+", expr)]
    pool = list(allowed)
    for n in used:
        if n not in pool:
            return None
        pool.remove(n)
    try:
        return float(eval(expr))
    except Exception:
        return None


# ── Reward functions ───────────────────────────────────────────────


def exact_match_reward(prompts: list, completions: list, target, numbers, **kwargs):
    """1.0 if the expression evaluates exactly to the target."""
    rewards = []
    for completion, tgt, nums in zip(completions, target, numbers):
        sol = parse_solution(completion[0]["content"])
        if not sol or any(p in sol.lower() for p in GIVE_UP_PHRASES):
            rewards.append(0.0)
            continue
        result = eval_expression(sol, nums)
        rewards.append(1.0 if result is not None and abs(result - tgt) < 1e-6 else 0.0)
    return rewards


def closeness_reward(prompts: list, completions: list, target, numbers, **kwargs):
    """Smooth reward: 0.5^(distance/10). Rewards near-misses."""
    rewards = []
    for completion, tgt, nums in zip(completions, target, numbers):
        sol = parse_solution(completion[0]["content"])
        if not sol or any(p in sol.lower() for p in GIVE_UP_PHRASES):
            rewards.append(0.0)
            continue
        result = eval_expression(sol, nums)
        if result is None:
            rewards.append(0.0)
        else:
            rewards.append(0.5 ** (abs(result - tgt) / 10))
    return rewards


def format_reward(prompts: list, completions: list, **kwargs):
    """1.0 if output has proper <reasoning>/<solution> XML tags with operators."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        has_tags = bool(
            re.search(r"<reasoning>.+?</reasoning>", content, re.DOTALL)
        ) and bool(re.search(r"<solution>.+?</solution>", content, re.DOTALL))
        sol = parse_solution(content)
        has_ops = bool(sol and re.search(r"[+\-*/]", sol))
        rewards.append(1.0 if has_tags and has_ops else 0.0)
    return rewards


# ── Training ───────────────────────────────────────────────────────


def main():
    train_ds = generate_puzzles(500, seed=42)
    eval_ds = generate_puzzles(100, seed=123)

    config = GRPOConfig(
        output_dir="countdown-grpo",
        max_steps=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_completion_length=512,
        learning_rate=1e-6,
        beta=0.0,
        temperature=1.0,
        reward_weights=[1.0, 0.3, 0.1],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1,
        log_completions=True,
        save_steps=25,
        eval_strategy="steps",
        eval_steps=25,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[exact_match_reward, closeness_reward, format_reward],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model("countdown-grpo/final")


if __name__ == "__main__":
    main()
