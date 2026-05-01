"""
triagerl.training.train
-----------------------
Small orchestration layer exposing `main()` which wires the
`TriageEpisodeDataset` and deterministic `build_reward_fn` into the
existing GRPO training loop. Keeps changes minimal: uses the same
external trainer setup but replaces the broken inline dataset/reward
with the robust implementations in `triagerl.training`.
"""
from __future__ import annotations

import logging
from typing import Callable

from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOConfig, GRPOTrainer

from triagerl.tasks.loader import load_all_tasks, get_task
from triagerl.training.dataset import TriageEpisodeDataset
from triagerl.training.reward_fn import build_reward_fn
from triagerl.env.triage_env import MedicalTriageEnv

PatchDPOTrainer()

logger = logging.getLogger(__name__)


def _env_factory(task_id: str) -> MedicalTriageEnv:
	# Wrap loader->env construction. The dataset and reward function
	# will ensure deterministic seeds via reset(seed=...).
	task = get_task(task_id)
	return MedicalTriageEnv(task_config=task)


def main():
	# Minimal parity with prior top-level script while using
	# deterministic dataset + reward_fn implementations.
	max_seq_length = 1536

	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name="unsloth/Llama-3-8b-Instruct-bnb-4bit",
		max_seq_length=max_seq_length,
		load_in_4bit=True,
		fast_inference=True,
	)

	model = FastLanguageModel.get_peft_model(
		model,
		r=16,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
						"gate_proj", "up_proj", "down_proj"],
		lora_alpha=16,
		lora_dropout=0,
		bias="none",
		use_gradient_checkpointing="unsloth",
		random_state=42,
	)

	task_map, task_ids = load_all_tasks()

	# Build deterministic on-policy dataset for epoch 0. Training
	# loop should call dataset.refresh(epoch) between epochs for
	# true on-policy behaviour; here we provide the initial dataset.
	dataset_obj = TriageEpisodeDataset(
		task_ids=task_ids,
		n_samples=128,
		global_seed=42,
		env_factory=_env_factory,
		include_mid_episode=True,
	)

	hf_dataset = dataset_obj.to_hf_dataset()

	reward_fn = build_reward_fn(task_map=task_map, env_factory=_env_factory, log_components=True)

	training_args = GRPOConfig(
		output_dir="outputs/triage_grpo",
		learning_rate=1e-5,
		per_device_train_batch_size=4,
		gradient_accumulation_steps=4,
		max_prompt_length=1024,
		max_completion_length=512,
		num_generations=4, # Group size for GRPO
		max_steps=200,
		logging_steps=10,
		save_steps=100,
	)

	trainer = GRPOTrainer(
		model=model,
		processing_class=tokenizer,
		reward_funcs=[reward_fn],
		args=training_args,
		train_dataset=hf_dataset,
	)

	# Epoched training loop with on-policy dataset refresh between epochs.
	# We divide the configured total `max_steps` into `num_epochs` slices and
	# call `dataset_obj.refresh(epoch)` before each epoch so that every epoch
	# trains on freshly generated episodes. The trainer is reused to preserve
	# optimizer state and checkpointing behaviour.
	num_epochs = 10
	total_steps = int(training_args.max_steps or 200)
	steps_per_epoch = max(1, total_steps // num_epochs)
	remaining = total_steps

	logger.info("Starting GRPO training with deterministic dataset and reward_fn")
	for epoch in range(num_epochs):
		this_epoch_steps = steps_per_epoch if epoch < num_epochs - 1 else remaining
		logger.info("Epoch %d/%d: refreshing dataset (seeded) and running %d steps", epoch + 1, num_epochs, this_epoch_steps)
		# Refresh dataset deterministically for this epoch
		dataset_obj.refresh(epoch=epoch)
		hf_dataset = dataset_obj.to_hf_dataset()
		# Update trainer dataset in-place so trainer uses fresh episodes
		trainer.train_dataset = hf_dataset
		# Run training for this epoch slice. Pass max_steps override to limit
		# the number of steps taken in this call. Many Trainer implementations
		# accept `max_steps` as an override; GRPOTrainer follows the HF API.
		trainer.train(max_steps=this_epoch_steps)
		remaining -= this_epoch_steps


	model.save_pretrained("outputs/triage_lora")
	tokenizer.save_pretrained("outputs/triage_lora")


if __name__ == "__main__":
	main()
