# How This Fine-Tuning Works

This project is doing **RL post-training** for a chat model on Countdown puzzles.
It is not supervised fine-tuning with target answers. The model learns from reward signals.

## 1) What data the model sees

- `generate_puzzles(...)` builds puzzle prompts and caches them in `puzzles.json`.
- Each row contains:
  - a system instruction,
  - a user puzzle (`Numbers: ...`, `Target: ...`),
  - metadata (`target`, `numbers`) used for reward calculation.

## 2) What the model outputs

For each prompt, the trainer samples multiple completions (`num_generations`).
The completion format is expected to include:

- `<reasoning>...</reasoning>`
- `<solution>...</solution>`

Only the `<solution>` part is parsed for arithmetic reward.

## 3) How rewards are computed

Three reward functions are used, then combined by `reward_weights=[1.0, 0.3, 0.1]`.

- `exact_match_reward`:
  - extracts `<solution>`,
  - verifies only allowed numbers are used (no reuse),
  - evaluates expression,
  - gives `1.0` only if result equals target.
- `closeness_reward`:
  - evaluates arithmetic (without number pool checks),
  - gives smooth reward `0.5 ** (distance / 10)`.
- `format_reward`:
  - `+0.5` if reasoning tags exist,
  - `+0.5` if solution looks like arithmetic and is not a give-up phrase.

## 4) How GRPO/CISPO updates happen

In each training step:

1. Sample prompts from the train dataset.
2. Generate multiple candidate completions per prompt.
3. Score each completion with the reward functions.
4. Convert rewards into relative advantages within each prompt group.
5. Update model log-probabilities with policy optimization loss.

Your config uses:

- `loss_type="cispo"` (CISPO variant),
- `importance_sampling_level="token"`,
- `epsilon_high=8.0`,
- `beta=0.0` (no KL penalty term),
- small batch/generation settings for memory safety.

So behavior is: promote higher-reward completions and suppress lower-reward ones, using the same prompt-level comparison group.

## 5) Why this is different from SFT

- SFT needs a gold answer and maximizes likelihood of that answer.
- This RL setup only needs a reward function.
- For puzzles, this is useful because many valid solutions exist.

## 6) Practical limits of this specific setup

- `eval(...)` is still used for arithmetic parsing. It is regex-gated but still brittle.
- `closeness_reward` can reward expressions that are near target but violate number-usage rules.
- `format_reward` can be gamed by valid tags + operator text without truly correct solving.

These are normal tradeoffs for a first-pass RL environment.
