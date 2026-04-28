import re
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from medical_triage_env.env import MedicalTriageEnv
from medical_triage_env.models import TriageAction

PatchDPOTrainer()

# ---------------------------------------------------------------------------
# Robust Regex JSON Extraction Utility
# ---------------------------------------------------------------------------
def extract_action_json(completion: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extracts a JSON object from a string that might contain markdown blocks
    or conversational preamble.
    """
    # Try to find a JSON code block
    json_block_match = re.search(r"```json\s*(.*?)\s*```", completion, re.DOTALL)
    if json_block_match:
        content = json_block_match.group(1)
    else:
        # Fallback: find anything that looks like a JSON dictionary
        dict_match = re.search(r"\{.*\}", completion, re.DOTALL)
        if dict_match:
            content = dict_match.group(0)
        else:
            content = completion

    try:
        return json.loads(content), ""
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

# ---------------------------------------------------------------------------
# GRPO Environment Reward Function
# ---------------------------------------------------------------------------
def triage_env_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    A stateless reward function for GRPOTrainer.
    In a real multi-step setup, you would manage episode state carefully.
    Here we evaluate the first step on a randomly assigned task.
    """
    rewards = []
    
    # Normally we'd extract the task_id from the prompt if it's encoded there,
    # but for this example we'll sample a task to evaluate the completion.
    from medical_triage_env.tasks import load_all_tasks
    _, task_ids = load_all_tasks()
    
    for completion in completions:
        task_id = random.choice(task_ids)
        # Using a curriculum dampening factor of 0.1 for early training exploration
        env = MedicalTriageEnv(task_id, clarify_penalty_dampening=0.1)
        env.reset()
        
        parsed_dict, err = extract_action_json(completion)
        if not parsed_dict:
            # Parsing completely failed
            rewards.append(-1.0)
            continue
            
        try:
            action = TriageAction.model_validate(parsed_dict)
        except Exception:
            # Schema validation failed
            rewards.append(-0.8)
            continue
            
        # Step the environment and get the reward
        try:
            _, reward, _, _ = env.step(action)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-1.0)
            
    return rewards

# ---------------------------------------------------------------------------
# Training Setup
# ---------------------------------------------------------------------------
def main():
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

    # Prepare a dummy dataset for the initial state of random tasks
    # (In a real setup, pre-generate these initial states)
    from medical_triage_env.tasks import load_all_tasks
    _, task_ids = load_all_tasks()
    
    prompts = []
    for _ in range(128):
        task_id = random.choice(task_ids)
        env = MedicalTriageEnv(task_id)
        obs = env.reset()
        obs_json = obs.model_dump_json()
        system_prompt = (
            "You are an emergency triage policy. Return strict JSON with keys: "
            "action_type, esi_level (for classify), clarifying_question (for clarify), "
            "reasoning, recommended_actions, confidence."
        )
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{obs_json}\n<|assistant|>\n"
        prompts.append({"prompt": prompt})
        
    dataset = Dataset.from_list(prompts)

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
        reward_funcs=[triage_env_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()
    
    # Save the trained LoRA adapters
    model.save_pretrained("outputs/triage_lora")
    tokenizer.save_pretrained("outputs/triage_lora")

if __name__ == "__main__":
    main()
