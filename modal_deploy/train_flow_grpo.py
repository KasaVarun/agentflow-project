"""
Flow-GRPO training script for AgentFlow planner on Modal.
Trains Qwen3.5-0.8B with LoRA using multi-turn GRPO.

Adapted from TinyZero-LoRA (github.com/KasaVarun/tinyzero-lora):
- LoRA config (r=16, alpha=64, q/k/v/o proj)
- Modal Volume pattern (volume.commit() after checkpoint)
- TRL GRPOConfig for hyperparameter reference

Key design (Flow-GRPO, AgentFlow paper):
- Planner (LoRA model) runs on GPU
- Fixed engine (executor/verifier/generator) calls Together AI
- G=8 trajectories sampled per query
- Binary trajectory reward broadcast to ALL planner turns
- Group-normalized advantages (mean/std across G)
- PPO-clipped policy gradient + KL penalty vs frozen reference
- Checkpoints saved to Modal Volume every CHECKPOINT_STEPS steps

Usage:
    modal run modal_deploy/train_flow_grpo.py
    modal run modal_deploy/train_flow_grpo.py --experiment-name flow_grpo_0.8b --max-steps 500

Cost: A10G ~$1.10/hr. 500 steps * 8 trajs * ~10s/traj = ~11 hrs = ~$12.
Add --benchmark text2sql to train for SQL task (Step 6).
"""
import modal

# ============================================================
# CONFIGURATION
# ============================================================
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
FIXED_ENGINE = "Qwen/Qwen2.5-7B-Instruct-Turbo"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

GPU_CONFIG = "A10G"
TRAINING_TIMEOUT = 72000    # 20 hours max
CHECKPOINT_STEPS = 50
MAX_STEPS = 500
GROUP_SIZE = 8              # G: trajectories per query (paper uses 8)
MAX_TURNS = 3               # Max planner turns per trajectory

# LoRA (from TinyZero-LoRA config)
LORA_R = 16
LORA_ALPHA = 64
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_DROPOUT = 0.05

# GRPO hyperparams (from TinyZero-LoRA config)
LEARNING_RATE = 1e-4
CLIP_EPS = 0.2
KL_COEF = 0.01
MAX_COMPLETION_TOKENS = 512
# ============================================================

app = modal.App("agentflow-train")

checkpoints_volume = modal.Volume.from_name("agentflow-checkpoints", create_if_missing=True)
results_volume = modal.Volume.from_name("agentflow-results", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "peft>=0.14.0",
        "trl>=0.12.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "openai>=1.50.0",
        "huggingface_hub>=0.25.0",
        "wikipedia>=1.4.0",
        "tqdm",
    )
    .run_commands(
        f"python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"{BASE_MODEL}\")'",
        secrets=[hf_secret],
    )
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_training_data(benchmark: str = "qa", max_samples: int = 5000):
    """
    Load training data.
    - benchmark="qa": NQ + HotpotQA mix (for paper benchmarks, Steps 5)
    - benchmark="sql": Spider train split (for Text-to-SQL, Step 6)
    """
    from datasets import load_dataset
    samples = []

    if benchmark == "sql":
        print("Loading Spider train split for Text-to-SQL training...")
        try:
            ds = load_dataset("spider")
            for ex in ds["train"]:
                db_id = ex["db_id"]
                question = ex["question"]
                gold_sql = ex["query"]
                query = (
                    f"Convert this question to SQL for the '{db_id}' database.\n"
                    f"Question: {question}\n"
                    f"Output the SQL in <answer> and </answer> tags."
                )
                samples.append({"query": query, "answer": gold_sql, "db_id": db_id})
                if len(samples) >= max_samples:
                    break
            print(f"Loaded {len(samples)} Spider train samples")
        except Exception as e:
            print(f"[warn] Could not load Spider: {e}")
        return samples

    # Default: QA mix (NQ + HotpotQA)
    try:
        nq = load_dataset("nq_open", split="train")
        for ex in nq:
            samples.append({"query": ex["question"], "answer": ex["answer"][0]})
            if len(samples) >= max_samples // 2:
                break
        print(f"Loaded {len(samples)} NQ samples")
    except Exception as e:
        print(f"[warn] NQ failed: {e}")

    try:
        hp = load_dataset("hotpot_qa", "distractor", split="train")
        count = 0
        for ex in hp:
            samples.append({"query": ex["question"], "answer": ex["answer"]})
            count += 1
            if len(samples) >= max_samples:
                break
        print(f"Loaded {count} HotpotQA samples")
    except Exception as e:
        print(f"[warn] HotpotQA failed: {e}")

    if not samples:
        print("[warn] Using fallback training data")
        samples = [
            {"query": "What is the capital of France?", "answer": "Paris"},
            {"query": "What is 2+2?", "answer": "4"},
        ] * 200

    return samples


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _judge_answer(predicted: str, gold, question: str, client) -> float:
    """Binary LLM-as-judge via Together AI. Returns 1.0 or 0.0."""
    import re
    gold_str = " OR ".join(str(a) for a in gold) if isinstance(gold, list) else str(gold)

    matches = re.findall(r"<answer>(.*?)</answer>", str(predicted), re.DOTALL)
    if matches:
        predicted = matches[-1].strip()

    prompt = (
        f"Is the predicted answer correct?\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_str}\n"
        f"Predicted answer: {predicted}\n\n"
        f"Respond with exactly '1' if correct, '0' if incorrect."
    )
    try:
        resp = client.chat.completions.create(
            model=FIXED_ENGINE,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        return 1.0 if "1" in resp.choices[0].message.content.strip() else 0.0
    except Exception as e:
        print(f"[judge error] {e}")
        return 0.0


# ---------------------------------------------------------------------------
# AgentFlow trajectory rollout
# ---------------------------------------------------------------------------

def _run_trajectory(query: str, model, tokenizer, together_client, device):
    """
    Run one AgentFlow trajectory with the LoRA planner.
    Returns (final_answer: str, planner_turns: list of (prompt_ids, response_ids))
    """
    import torch
    import json
    import wikipedia

    memory = []
    planner_turns = []

    def call_planner(prompt_text: str):
        fmt = (
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(
            fmt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_COMPLETION_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        response_ids = out[0][prompt_len:]
        planner_turns.append((inputs["input_ids"][0].cpu(), response_ids.cpu()))
        return tokenizer.decode(response_ids, skip_special_tokens=True)

    def call_fixed(prompt: str, max_tokens: int = 256) -> str:
        try:
            resp = together_client.chat.completions.create(
                model=FIXED_ENGINE,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[error: {e}]"

    def wiki_search(q: str) -> str:
        try:
            results = wikipedia.search(q, results=3)
            if not results:
                return "No results."
            page = wikipedia.page(results[0], auto_suggest=False)
            return page.content[:1500]
        except Exception as e:
            return f"Wikipedia error: {e}"

    # Agentic loop
    for turn in range(MAX_TURNS):
        mem_str = "\n".join(
            f"Step {i+1} [{t}]: {r[:150]}" for i, (t, r) in enumerate(memory)
        ) if memory else "None"

        plan_prompt = (
            f"Query: {query}\n"
            f"Previous steps:\n{mem_str}\n\n"
            f"Available tools: Wikipedia_Search, Direct_Answer\n"
            f"Choose the best next action. Respond in JSON:\n"
            f'{{\"tool\": \"Wikipedia_Search\" or \"Direct_Answer\", '
            f'\"sub_goal\": \"<search query or answer>\"}}'
        )
        plan_response = call_planner(plan_prompt)

        tool, sub_goal = "Direct_Answer", query
        try:
            data = json.loads(plan_response.strip())
            tool = data.get("tool", "Direct_Answer")
            sub_goal = data.get("sub_goal", query)
        except Exception:
            pass

        result = wiki_search(sub_goal) if tool == "Wikipedia_Search" else call_fixed(
            f"Answer step by step: {query}\nFocus on: {sub_goal}"
        )
        memory.append((tool, result))

        verdict = call_fixed(
            f"Do we have enough info to answer: {query}\n"
            f"Info gathered: {result[:400]}\n"
            f"Reply STOP or CONTINUE.",
            max_tokens=10,
        )
        if "STOP" in verdict.upper():
            break

    # Generator
    ctx = "\n".join(f"- {r[:300]}" for _, r in memory)
    final = call_fixed(
        f"Answer concisely based on research.\n"
        f"Question: {query}\nResearch:\n{ctx}\n\n"
        f"Provide the answer in <answer> and </answer> tags.",
        max_tokens=150,
    )
    return final, planner_turns


# ---------------------------------------------------------------------------
# Flow-GRPO loss
# ---------------------------------------------------------------------------

def _compute_flow_grpo_loss(model, ref_model, trajectories, rewards, device):
    """
    Flow-GRPO loss (AgentFlow paper, Section 3.2):
    1. Normalize rewards across G trajectories -> advantages
    2. For each planner turn in each trajectory:
       - Compute log-ratio π_θ / π_ref (importance weight)
       - PPO-clipped objective scaled by advantage
       - KL penalty
    3. Average over all response tokens
    """
    import torch
    import torch.nn.functional as F

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std().clamp(min=1e-8)
    advantages = (rewards_t - mean_r) / std_r  # [G]

    total_loss = torch.zeros(1, device=device, requires_grad=False)
    n_tokens = 0

    for traj_idx, (_, planner_turns) in enumerate(trajectories):
        adv = advantages[traj_idx]

        for prompt_ids, response_ids in planner_turns:
            if len(response_ids) == 0:
                continue

            full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
            prompt_len = prompt_ids.shape[0]
            resp_len = response_ids.shape[0]

            # Current model log-probs
            with torch.enable_grad():
                out = model(input_ids=full_ids, use_cache=False)
            logits = out.logits[0, prompt_len - 1:prompt_len + resp_len - 1]  # [resp_len, vocab]
            log_probs = F.log_softmax(logits, dim=-1)
            resp_ids_dev = response_ids.to(device)
            tok_log_probs = log_probs.gather(1, resp_ids_dev.unsqueeze(1)).squeeze(1)  # [resp_len]

            # Reference model log-probs (frozen)
            with torch.no_grad():
                ref_out = ref_model(input_ids=full_ids, use_cache=False)
            ref_logits = ref_out.logits[0, prompt_len - 1:prompt_len + resp_len - 1]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_tok_log_probs = ref_log_probs.gather(1, resp_ids_dev.unsqueeze(1)).squeeze(1)

            # Log importance ratio
            log_ratio = tok_log_probs - ref_tok_log_probs.detach()  # [resp_len]
            ratio = log_ratio.exp()

            # PPO-clipped objective
            pg1 = ratio * adv
            pg2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            pg_loss = -torch.min(pg1, pg2).mean()

            # KL penalty (forward KL: π_θ log(π_θ/π_ref))
            kl = (tok_log_probs.exp() * log_ratio).mean()
            kl_loss = KL_COEF * kl

            turn_loss = (pg_loss + kl_loss).unsqueeze(0)
            total_loss = total_loss + turn_loss
            n_tokens += resp_len

    if n_tokens > 0:
        total_loss = total_loss / n_tokens
    return total_loss.squeeze()


# ---------------------------------------------------------------------------
# Modal training function
# ---------------------------------------------------------------------------

@app.function(
    gpu=GPU_CONFIG,
    image=train_image,
    timeout=TRAINING_TIMEOUT,
    scaledown_window=300,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/results": results_volume,
    },
    secrets=[modal.Secret.from_name("together-ai"), hf_secret],
)
def train(
    experiment_name: str = "flow_grpo_0.8b",
    max_steps: int = MAX_STEPS,
    resume_from_checkpoint: bool = True,
    benchmark: str = "qa",  # "qa" or "sql"
):
    """
    Flow-GRPO training. Saves to Modal Volumes every CHECKPOINT_STEPS.
    Set benchmark="sql" to train for Text-to-SQL (Step 6).
    """
    import os
    import json
    import random
    import torch
    from openai import OpenAI
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {BASE_MODEL} | Benchmark: {benchmark}")

    together_api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_API_KEY_SECRET")
    if not together_api_key:
        # Print all env vars with 'TOGETHER' to help debug secret key name
        keys = [k for k in os.environ if "TOGETHER" in k.upper()]
        raise ValueError(f"TOGETHER_API_KEY not found. Available TOGETHER* vars: {keys}")
    together_client = OpenAI(api_key=together_api_key, base_url=TOGETHER_BASE_URL)

    checkpoint_dir = f"/checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(checkpoint_dir, "latest")
    meta_file = os.path.join(checkpoint_dir, "meta.json")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (resume or fresh LoRA)
    start_step = 0
    if resume_from_checkpoint and os.path.exists(latest_ckpt):
        print(f"Resuming from: {latest_ckpt}")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, latest_ckpt).to(device)
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                start_step = json.load(f).get("step", 0)
        print(f"Resuming from step {start_step}")
    else:
        print("Initializing fresh LoRA model...")
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, lora_cfg).to(device)
        model.print_trainable_parameters()

    # Frozen reference model (base weights only, no LoRA)
    print("Loading frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01,
    )

    # Training data
    print("Loading training data...")
    training_data = _load_training_data(benchmark=benchmark, max_samples=5000)
    random.shuffle(training_data)
    print(f"Total training samples: {len(training_data)}")

    # Training loop
    log = []
    print(f"\nFlow-GRPO: steps {start_step} -> {max_steps}, G={GROUP_SIZE}\n")

    for step in range(start_step, max_steps):
        sample = training_data[step % len(training_data)]
        query = sample["query"]
        gold = sample["answer"]

        # Sample G trajectories
        trajectories, rewards = [], []
        for g in range(GROUP_SIZE):
            try:
                ans, turns = _run_trajectory(query, model, tokenizer, together_client, device)
                r = _judge_answer(ans, gold, query, together_client)
            except Exception as e:
                print(f"  [traj {g} error] {e}")
                ans, turns, r = "", [], 0.0
            trajectories.append((ans, turns))
            rewards.append(r)

        mean_r = sum(rewards) / len(rewards)

        # Skip step if all rewards identical (no learning signal)
        if len(set(rewards)) == 1:
            print(f"Step {step+1}/{max_steps} | skip (uniform rewards={rewards[0]})")
            log.append({"step": step + 1, "loss": None, "mean_reward": mean_r, "skipped": True})
            continue

        # Flow-GRPO update
        optimizer.zero_grad()
        loss = _compute_flow_grpo_loss(model, ref_model, trajectories, rewards, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=0.5
        )
        optimizer.step()

        loss_val = float(loss.item())
        log.append({"step": step + 1, "loss": loss_val, "mean_reward": mean_r, "rewards": rewards})
        print(f"Step {step+1}/{max_steps} | loss={loss_val:.4f} | reward={mean_r:.2f} | {rewards}")

        # Checkpoint
        if (step + 1) % CHECKPOINT_STEPS == 0 or (step + 1) == max_steps:
            print(f"  Saving checkpoint @ step {step+1}...")
            model.save_pretrained(latest_ckpt)
            tokenizer.save_pretrained(latest_ckpt)
            model.save_pretrained(os.path.join(checkpoint_dir, f"step_{step+1}"))
            with open(meta_file, "w") as f:
                json.dump({"step": step + 1, "model": BASE_MODEL, "benchmark": benchmark}, f)
            with open(f"/results/{experiment_name}_log.json", "w") as f:
                json.dump(log, f, indent=2)
            checkpoints_volume.commit()
            results_volume.commit()
            print(f"  Checkpoint saved and committed.")

    print(f"\nTraining complete. Checkpoint: {latest_ckpt}")
    return {
        "experiment": experiment_name,
        "steps": max_steps,
        "final_reward": log[-1]["mean_reward"] if log else 0,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    experiment_name: str = "flow_grpo_0.8b",
    max_steps: int = MAX_STEPS,
    resume: bool = True,
    benchmark: str = "qa",
):
    """
    Launch Flow-GRPO training on Modal.

    Steps 5 (paper benchmarks):
        modal run modal_deploy/train_flow_grpo.py

    Step 6 (Text-to-SQL):
        modal run modal_deploy/train_flow_grpo.py --benchmark sql --experiment-name flow_grpo_sql_0.8b

    Prerequisites:
        modal secret create together-api-key TOGETHER_API_KEY=<your-key>
    """
    print("=" * 60)
    print("Flow-GRPO Training")
    print("=" * 60)
    print(f"Experiment:  {experiment_name}")
    print(f"Model:       {BASE_MODEL}")
    print(f"GPU:         {GPU_CONFIG}")
    print(f"Max steps:   {max_steps}")
    print(f"Group size:  {GROUP_SIZE}")
    print(f"Benchmark:   {benchmark}")
    print(f"Resume:      {resume}")
    print()

    # Cost estimate
    hrs = max_steps * GROUP_SIZE * MAX_TURNS * 10 / 3600
    cost = hrs * 1.10
    print(f"Cost estimate: ~{hrs:.1f} GPU-hrs @ $1.10/hr = ~${cost:.0f}")
    print()

    result = train.remote(
        experiment_name=experiment_name,
        max_steps=max_steps,
        resume_from_checkpoint=resume,
        benchmark=benchmark,
    )
    print(f"\nResult: {result}")
