# Countdown GRPO with TRL + vLLM Serve

This branch keeps the repository focused on the stock `trl vllm-serve`
workflow. A dedicated helper script starts vLLM on eight H100 GPUs, and
`run.sh` launches GRPO fine-tuning via `torchrun` against that server.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

(Use `uv pip install -e .` if you prefer `uv`.)

## 1. Start vLLM with TRL's CLI

Run the helper, which simply wraps `trl vllm-serve` with sensible
defaults for 8×H100 80GB:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/launch_vllm_server.sh
```

Important flags (override via env vars):

- `MODEL` – defaults to `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `TP_SIZE` – tensor parallel degree (defaults to GPU count)
- `HOST`/`PORT` – bind address (`0.0.0.0:8000` by default)
- `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS`,
  `GPU_MEMORY_UTILIZATION` – tuned for 32–64 concurrent sequences so
  the RL rollout pressure stays within 80 GB per device.

The script exports `CUDA_VISIBLE_DEVICES`, prints the derived settings,
then execs `trl vllm-serve` so logs remain in your terminal.

## 2. Launch GRPO training

`run.sh` now targets `countdown_trl_vllm.py` and assumes eight GPUs. It
also checks the vLLM health endpoint before calling `torchrun`:

```bash
./run.sh
```

Key knobs:

- `TRAIN_GPUS` – CSV list passed to `CUDA_VISIBLE_DEVICES`
- `TRAIN_NPROC` – override world size if you need a subset of devices
- `VLLM_SERVER_URL` – change if the server listens elsewhere
- `TRAIN_SCRIPT` – swap in a different script if desired

You can still pass extra args (currently unused) after `./run.sh` and
they will be forwarded to the Python script.

## Countdown configuration

`countdown_trl_vllm.py` exposes a few environment-driven settings so you
can match the training load to your cluster:

| Variable | Default | Description |
| --- | --- | --- |
| `COUNTDOWN_TRAIN_PUZZLES` | 1024 | Generated puzzles for training |
| `COUNTDOWN_EVAL_PUZZLES` | 256 | Eval puzzles |
| `COUNTDOWN_MAX_STEPS` | 400 | GRPO steps |
| `COUNTDOWN_PER_DEVICE_BATCH` | 1 | Prompts per GPU per micro-batch |
| `COUNTDOWN_GRAD_ACCUM` | 4 | Accumulation steps (global prompts = batch × grad × world size) |
| `COUNTDOWN_NUM_GENERATIONS` | 4 | Completions sampled per prompt |
| `COUNTDOWN_MAX_COMPLETION_LENGTH` | 192 | Tokens per response |
| `COUNTDOWN_LEARNING_RATE` | 8e-7 | LR for LoRA adapters |
| `COUNTDOWN_OUTPUT_DIR` | `countdown-grpo` | Checkpoints/logs |
| `VLLM_SERVER_URL` | `http://127.0.0.1:8000` | Where `trl vllm-serve` listens |

On an 8×H100 rig the defaults result in 32 prompts (and 128
completions) per optimizer step, which keeps the server well within the
configured 64-sequence cap while leaving memory headroom for optimizer
state and activations.

## Outputs

- Checkpoints land in `$COUNTDOWN_OUTPUT_DIR` (final weights saved to
  `<output_dir>/final`).
- `reward_curve.png` – saved after training if rewards exist.

Happy puzzling!
