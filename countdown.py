"""
Countdown Numbers Game - GRPO Training with TRL

Trains a language model to solve arithmetic countdown puzzles using
Group Relative Policy Optimization (GRPO).

Reproduction of: https://samikhan.ai/blog/countdown-rl.html
"""

import random
import re
import json
import os

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen3-4B"

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
SMALL_NUMBERS = list(range(1, 11)) * 2
PUZZLES_FILE = os.path.join(os.path.dirname(__file__), "puzzles.json")


# ── Dataset ────────────────────────────────────────────────────────


def sample_numbers(rng: random.Random) -> list[int]:
    """Sample numbers like the real Countdown game: 1-3 large + rest small."""
    num_large = rng.randint(1, 3)
    large = rng.sample(LARGE_NUMBERS, num_large)
    small = rng.sample(SMALL_NUMBERS, 6 - num_large)
    numbers = large + small
    rng.shuffle(numbers)
    return numbers


def puzzle_row(numbers: list[int], target: int) -> dict:
    """Create a training row in TRL chat format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Numbers: {', '.join(map(str, numbers))}\nTarget: {target}",
            },
        ],
        "target": target,
        "numbers": numbers,
    }


def load_puzzle_cache() -> dict[str, list[dict]]:
    """Load puzzle cache from disk (expects valid schema)."""
    if not os.path.exists(PUZZLES_FILE):
        return {}
    try:
        with open(PUZZLES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    return raw.get("by_seed", raw)


def save_puzzle_cache(cache: dict[str, list[dict]]) -> None:
    """Persist puzzle cache to disk."""
    payload = {"version": 1, "by_seed": cache}
    with open(PUZZLES_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_target(rng: random.Random, numbers: list[int]) -> int | None:
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: a / b if b != 0 and a % b == 0 else None),
    ]
    pool = list(numbers)
    rng.shuffle(pool)

    # how many of the numbers to use
    k = rng.randint(2, min(4, len(pool)))
    result = pool[0]
    for i in range(1, k):
        _, fn = rng.choice(ops)
        new_result = fn(result, pool[i])
        if new_result is None: return None
        result = int(new_result)
    return result if 100 <= result <= 999 else None


def build_puzzle(rng: random.Random) -> dict | None:
    """Generate one puzzle candidate using current source-number and target logic."""
    numbers = sample_numbers(rng)
    target = build_target(rng, numbers)
    if target is None:
        return None
    return {"numbers": numbers, "target": target}


def generate_puzzles(n: int, seed: int = 42) -> Dataset:
    """Generate n solvable countdown puzzles by constructing solutions first."""
    cache = load_puzzle_cache()
    seed_key = str(seed)
    cached = cache.setdefault(seed_key, [])

    if len(cached) < n:
        # Keep your generation logic, only appending what is missing.
        rng = random.Random(seed + len(cached))
        seen = {(tuple(item["numbers"]), item["target"]) for item in cached}

        while len(cached) < n:
            puzzle = build_puzzle(rng)
            if puzzle is None:
                continue
            key = (tuple(puzzle["numbers"]), puzzle["target"])
            if key in seen:
                continue
            seen.add(key)
            cached.append(puzzle)

        save_puzzle_cache(cache)

    rows = [puzzle_row(item["numbers"], item["target"]) for item in cached[:n]]
    return Dataset.from_list(rows)


# ── Reward helpers ─────────────────────────────────────────────────

def get_completion_content(completion):
    """Extract content from a TRL completion payload."""
    return completion[0]["content"]

def parse_solution(text: str) -> str | None:
    m = SOLUTION_RE.search(text)
    return m.group(1).strip() if m else None


def is_give_up(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in GIVE_UP_PHRASES)


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


def eval_arithmetic(expr: str) -> float | None:
    """Evaluate arithmetic expression without number-pool checks."""
    if not re.fullmatch(r"[\d+\-*/().\s]+", expr):
        return None
    try:
        return float(eval(expr))
    except Exception:
        return None


# ── Reward functions ───────────────────────────────────────────────


def exact_match_reward(prompts: list, completions: list, target, numbers, **kwargs):
    """1.0 if the expression evaluates exactly to the target."""
    rewards = []
    for completion, tgt, nums in zip(completions, target, numbers):
        content = get_completion_content(completion)
        sol = parse_solution(content)
        if not sol or is_give_up(sol):
            rewards.append(0.0)
            continue
        result = eval_expression(sol, nums)
        rewards.append(1.0 if result is not None and abs(result - tgt) < 1e-6 else 0.0)
    return rewards


def closeness_reward(prompts: list, completions: list, target, numbers, **kwargs):
    """Smooth reward: 0.5^(distance/10). Rewards near-misses."""
    rewards = []
    for completion, tgt, nums in zip(completions, target, numbers):
        content = get_completion_content(completion)
        sol = parse_solution(content)
        if not sol or is_give_up(sol):
            rewards.append(0.0)
            continue
        result = eval_arithmetic(sol)
        if result is None:
            rewards.append(0.0)
        else:
            rewards.append(0.5 ** (abs(result - tgt) / 10))
    return rewards


def format_reward(prompts: list, completions: list, **kwargs):
    """Partial format reward: 0.5 for reasoning + 0.5 for valid-looking solution."""
    rewards = []
    for completion in completions:
        content = get_completion_content(completion)
        score = 0.0
        has_reasoning = bool(re.search(r"<reasoning>.+?</reasoning>", content, re.DOTALL))
        if has_reasoning:
            score += 0.5

        sol = parse_solution(content)
        if sol and not is_give_up(sol):
            has_ops = bool(re.search(r"[+\-*/]", sol))
            if has_ops:
                score += 0.5
        rewards.append(score)
    return rewards


# ── Training ───────────────────────────────────────────────────────


def main():
    train_ds = generate_puzzles(500, seed=42)
    eval_ds = generate_puzzles(100, seed=123)

    config = GRPOConfig(
        output_dir="countdown-grpo",
        max_steps=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # Match blog run: batch_size=128 rollouts_per_example=4.
        generation_batch_size=128,
        num_generations=4,
        num_generations_eval=4,
        max_completion_length=512,
        learning_rate=1e-6,
        beta=0.0,
        temperature=1.0,
        # CISPO-style objective: detach clipped IS ratio as a coefficient on log-prob.
        loss_type="cispo",
        # Use a wider upper clamp to mimic "stable off-policy-ish" settings from async stacks.
        epsilon_high=8.0,
        importance_sampling_level="token",
        use_vllm=True,
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.3,
        reward_weights=[1.0, 0.3, 0.1],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1,
        log_completions=True,
        save_steps=25,
        eval_strategy="steps",
        eval_steps=25,
        report_to="tensorboard",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[exact_match_reward, closeness_reward, format_reward],
        args=config,
        peft_config=peft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model("countdown-grpo/final")

    # ── Visualization ──────────────────────────────────────────────────
    # Plot reward over time
    log_history = trainer.state.log_history
    steps = [x["step"] for x in log_history if "reward" in x]
    rewards = [x["reward"] for x in log_history if "reward" in x]

    if rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, rewards, label="Average Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Training Reward Progress")
        plt.grid(True)
        plt.savefig("reward_curve.png")
        print("Saved training plot to reward_curve.png")


if __name__ == "__main__":
    main()
