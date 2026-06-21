"""
triagerl.training.train
-----------------------
Scaled-up GRPO training orchestrator for TriageRL OpenEnv benchmark.

Scale-up changes vs v0.1
------------------------
Model        : unsloth/Llama-3.1-8B-Instruct-bnb-4bit  (same family, improved checkpoint)
               → Switch to unsloth/Llama-3.3-70B-Instruct-bnb-4bit for full-scale training
LoRA rank    : r=16  →  r=64  (4× capacity; alpha matched for stable training)
Dataset      : 128 samples/epoch  →  512 samples/epoch
Steps        : 200 total  →  2000 total  (10× more gradient steps)
Epochs       : 10  →  20  (with on-policy dataset refresh each epoch)
Batch        : per_device=4, accum=4 (eff=16)  →  per_device=8, accum=8 (eff=64)
Seq length   : 1536  →  3072  (covers longer multi-step episodes and new task categories)
Completions  : 512  →  1024  (richer per-step reasoning)
GRPO groups  : 4  →  8  (better advantage estimation across more generations)
LR           : 1e-5  →  5e-6  (lower for larger LoRA; avoid over-shooting)
Curriculum   : clarify_penalty_dampening ramped 0.0→1.0 linearly over epochs so early
               exploration is unrestricted and later training applies full penalties.
Task corpus  : 20 tasks (5 categories)  →  35 tasks (9 categories including pediatric,
               toxicology, obstetric, trauma) — all reflected in dataset sampling.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOConfig, GRPOTrainer

from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.training.dataset import TriageEpisodeDataset
from triagerl.training.reward_fn import build_reward_fn
from triagerl.env.triage_env import MedicalTriageEnv

PatchDPOTrainer()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter constants — all configurable in one place
# ---------------------------------------------------------------------------

# Model selection
MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
# Uncomment for full-scale 70B training (requires multi-GPU setup):
# MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

# Sequence lengths
MAX_SEQ_LENGTH    = 3072   # was 1536 — covers longer multi-step episodes
MAX_PROMPT_LENGTH = 2048   # was 1024
MAX_COMP_LENGTH   = 1024   # was 512

# LoRA configuration
LORA_RANK     = 64    # was 16 — 4× capacity for richer task set
LORA_ALPHA    = 64    # match rank for unit-scale signal
LORA_DROPOUT  = 0.05  # small regularisation for larger rank

# Training scale
GLOBAL_SEED          = 42
N_SAMPLES_PER_EPOCH  = 512    # was 128 — 4× more on-policy episodes per epoch
NUM_EPOCHS           = 20     # was 10 — more curriculum coverage
TOTAL_STEPS          = 2000   # was 200 — 10× more gradient updates
NUM_GENERATIONS      = 8      # was 4 — better GRPO advantage estimates
LEARNING_RATE        = 5e-6   # was 1e-5 — lower for larger LoRA stability
PER_DEVICE_BATCH     = 8      # was 4
GRAD_ACCUM_STEPS     = 8      # was 4 — effective batch = 64
LOGGING_STEPS        = 25
SAVE_STEPS           = 250

# Curriculum dampening ramp: starts at 0.0 (free exploration) and reaches 1.0
# (full penalties) by the midpoint of training, then stays at 1.0.
# This prevents premature episode termination killing early GRPO exploration.
DAMPENING_START      = 0.0
DAMPENING_END        = 1.0
DAMPENING_RAMP_EPOCHS = NUM_EPOCHS // 2   # ramp over first half of training


def _compute_dampening(epoch: int) -> float:
    """Linearly ramp clarify_penalty_dampening from 0.0 to 1.0 over first half of epochs."""
    if epoch >= DAMPENING_RAMP_EPOCHS:
        return DAMPENING_END
    return DAMPENING_START + (DAMPENING_END - DAMPENING_START) * (
        epoch / DAMPENING_RAMP_EPOCHS
    )


def _env_factory(task_id: str, epoch: int = 0) -> MedicalTriageEnv:
    """
    Construct a MedicalTriageEnv with epoch-appropriate curriculum dampening.

    clarify_penalty_dampening ramps from 0.0 (no loop penalties — free exploration)
    to 1.0 (full penalties — curriculum pressure) linearly over the first half of
    training epochs. This prevents premature episode termination killing early
    GRPO generation diversity.
    """
    task = get_task(task_id)
    dampening = _compute_dampening(epoch)
    return MedicalTriageEnv(
        task_config=task,
        clarify_penalty_dampening=dampening,
    )


def _make_epoch_env_factory(epoch: int) -> Callable[[str], MedicalTriageEnv]:
    """Return an env_factory closure bound to a specific epoch for dampening."""
    def factory(task_id: str) -> MedicalTriageEnv:
        return _env_factory(task_id, epoch=epoch)
    return factory


def main():
    """
    Full-scale GRPO training loop with:
    - On-policy dataset refresh each epoch
    - Linearly ramped clarify_penalty_dampening
    - 35-task corpus across 9 medical categories
    """
    logger.info(
        "Starting TriageRL scaled training: model=%s, lora_r=%d, "
        "n_samples=%d, total_steps=%d, epochs=%d",
        MODEL_NAME, LORA_RANK, N_SAMPLES_PER_EPOCH, TOTAL_STEPS, NUM_EPOCHS,
    )

    # ── Model and tokenizer ─────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=GLOBAL_SEED,
    )

    # ── Task corpus ─────────────────────────────────────────────────────────
    task_map, task_ids = load_all_tasks()
    logger.info("Loaded %d tasks: %s", len(task_ids), sorted(task_ids))

    # Build difficulty map for curriculum-aware dataset sampling
    task_difficulty_map = {
        tid: task_map[tid].difficulty
        for tid in task_ids
        if tid in task_map
    }

    # ── Initial dataset (epoch 0 — no dampening, free exploration) ─────────
    epoch_0_factory = _make_epoch_env_factory(epoch=0)
    dataset_obj = TriageEpisodeDataset(
        task_ids=task_ids,
        n_samples=N_SAMPLES_PER_EPOCH,
        global_seed=GLOBAL_SEED,
        env_factory=epoch_0_factory,
        task_difficulty_map=task_difficulty_map,
        include_mid_episode=True,
    )
    hf_dataset = dataset_obj.to_hf_dataset()

    # ── Reward function ─────────────────────────────────────────────────────
    reward_fn = build_reward_fn(
        task_map=task_map,
        env_factory=epoch_0_factory,
        log_components=True,
    )

    # ── Training configuration ──────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir="outputs/triage_grpo_v2",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMP_LENGTH,
        num_generations=NUM_GENERATIONS,
        max_steps=TOTAL_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        warmup_steps=max(1, TOTAL_STEPS // 20),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        dataloader_num_workers=0,
        report_to="none",  # set to "wandb" if wandb is configured
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=hf_dataset,
    )

    # ── Epoched training loop with on-policy refresh ────────────────────────
    steps_per_epoch = max(1, TOTAL_STEPS // NUM_EPOCHS)
    remaining_steps = TOTAL_STEPS

    logger.info(
        "Beginning %d-epoch GRPO training loop. %d steps/epoch. "
        "Dampening ramp: %.1f→%.1f over first %d epochs.",
        NUM_EPOCHS, steps_per_epoch,
        DAMPENING_START, DAMPENING_END, DAMPENING_RAMP_EPOCHS,
    )

    for epoch in range(NUM_EPOCHS):
        dampening = _compute_dampening(epoch)
        this_epoch_steps = (
            steps_per_epoch if epoch < NUM_EPOCHS - 1 else remaining_steps
        )

        logger.info(
            "Epoch %d/%d | steps=%d | remaining=%d | dampening=%.2f",
            epoch + 1, NUM_EPOCHS, this_epoch_steps, remaining_steps, dampening,
        )

        # Rebuild env_factory with current epoch's dampening value
        epoch_factory = _make_epoch_env_factory(epoch=epoch)

        # Refresh dataset with this epoch's env factory (on-policy)
        dataset_obj._env_factory = epoch_factory
        dataset_obj.refresh(epoch=epoch)
        hf_dataset = dataset_obj.to_hf_dataset()

        # Rebuild reward function with updated dampening env factory
        reward_fn = build_reward_fn(
            task_map=task_map,
            env_factory=epoch_factory,
            log_components=True,
        )

        trainer.train_dataset = hf_dataset
        trainer.reward_funcs   = [reward_fn]

        trainer.train(max_steps=this_epoch_steps)
        remaining_steps -= this_epoch_steps

        logger.info(
            "Epoch %d/%d complete. Remaining steps: %d",
            epoch + 1, NUM_EPOCHS, remaining_steps,
        )

    # ── Save final model ────────────────────────────────────────────────────
    logger.info("Saving LoRA adapter to outputs/triage_lora_v2")
    model.save_pretrained("outputs/triage_lora_v2")
    tokenizer.save_pretrained("outputs/triage_lora_v2")
    logger.info("Training complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    main()
