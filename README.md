# AgentFlow: Self-Improving AI via Flow-GRPO

Northeastern University — Self-Improving AI Systems (Final Project, April 2026)

Reimplementation of [AgentFlow](https://arxiv.org/abs/2501.12599) (Huang et al., 2025): a multi-agent framework that trains a planner LLM to orchestrate tools using Flow-GRPO, a multi-turn extension of Group Relative Policy Optimization.

---

## Architecture

```
Query
  └─> Planner (trainable LLM) ──> tool calls ──> Executor
                                                      └─> Verifier
                                                               └─> Generator ──> Answer
```

Four specialized modules share a memory buffer:
- **Planner**: Decides which tool to call and what sub-goal to pursue. Only this module is trained.
- **Executor**: Runs the selected tool (Google Search, Wikipedia, SQL, Python).
- **Verifier**: Determines whether enough information has been gathered to stop.
- **Generator**: Produces the final answer from accumulated context.

---

## Results

### Step 1-2: AgentFlow Baseline (Qwen2.5-7B-Instruct)

| Benchmark | Ours | Paper Target |
|-----------|------|-------------|
| Bamboogle | **67.74%** | 58.4% |
| HotpotQA  | 49.48% | 51.3% |
| Musique   | 17.44% | 19.2% |
| 2WikiMultiHop | 37.89% | 60.0% |
| GAIA      | **17.36%** | 17.2% |

Bamboogle and GAIA match or exceed paper targets. 2Wiki is below target — comparative-fact questions requiring multi-hop entity lookup remain challenging.

### Step 3: Model Scaling Study (Qwen2.5 family)

| Model | Bamboogle | HotpotQA | Musique | 2Wiki | GAIA |
|-------|-----------|----------|---------|-------|------|
| 0.5B  | 37.1% | 27.6% | 6.1%  | 24.9% | 12.3% |
| 3B    | 34.7% | 28.6% | 7.1%  | 23.8% | 11.4% |
| 7B    | **67.7%** | **49.5%** | **17.4%** | **37.9%** | **17.4%** |

Clear scaling benefit at 7B. The 0.5B→3B gap is small; 3B→7B is dramatic.

### Step 4: Text-to-SQL (Spider Dev, Qwen2.5-7B)

Evaluated on 1034 Spider dev questions using execution accuracy.
**Result: 91.01% (941/1034)** — competitive with SOTA on Spider dev leaderboard.

### Step 5-6: Flow-GRPO Training

- **Step 5**: 500-step Flow-GRPO on Qwen2.5-0.5B-Instruct (QA tasks: NQ + HotpotQA)
  - LoRA: r=16, alpha=64, target modules: q/k/v/o_proj
  - Group size G=8, PPO clip ε=0.2, KL coef=0.01
  - Checkpoint: Modal Volume `agentflow-checkpoints/flow_grpo_0.5b`

- **Step 6**: 500-step Flow-GRPO on Qwen2.5-0.5B-Instruct (Spider SQL)
  - Same LoRA config, benchmark="sql"
  - Checkpoint: Modal Volume `agentflow-checkpoints/flow_grpo_sql_0.5b`

---

## Setup

### Prerequisites
- Python 3.11+
- Modal account (for GPU deployment)
- Together AI API key (for serverless LLM inference)
- Serper.dev API key (for Google Search)

### Installation

```bash
git clone https://github.com/KasaVarun/agentflow-project
cd agentflow-project
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```
TOGETHER_API_KEY=your_key
SERPER_API_KEY=your_key
PLANNER_BASE_URL=https://varunkasa77--agentflow-planner-serve.modal.run/v1
```

### Run a Benchmark

```bash
# Step 1-2: Baseline with 7B via Together AI
python scripts/run_benchmark.py \
    --benchmark bamboogle \
    --planner_engine "together-Qwen/Qwen2.5-7B-Instruct-Turbo" \
    --run_all --max_steps 10

# Evaluate results
python scripts/evaluate.py \
    --benchmark bamboogle \
    --result_dir results/bamboogle/together_Qwen_Qwen2.5_7B_Instruct_Turbo \
    --response_type direct_output

# Text-to-SQL (Step 4)
python scripts/run_benchmark.py \
    --benchmark text2sql \
    --planner_engine "together-Qwen/Qwen2.5-7B-Instruct-Turbo" \
    --exp_name step4_text2sql_7b \
    --run_all --max_steps 5

# Show all results
python scripts/show_results.py
```

### Deploy on Modal

```bash
# Deploy 7B planner
PYTHONUTF8=1 modal deploy modal_deploy/serve_planner.py

# Run Flow-GRPO training (Step 5)
PYTHONUTF8=1 modal run modal_deploy/train_flow_grpo.py \
    --experiment-name flow_grpo_0.5b --max-steps 500

# Deploy trained checkpoint
PYTHONUTF8=1 modal deploy modal_deploy/serve_grpo_checkpoint.py
```

---

## Project Structure

```
agentflow-project/
├── agentflow/              # Core framework
│   ├── planner/            # Planner module
│   ├── executor/           # Tool execution
│   ├── verifier/           # Stop condition
│   ├── generator/          # Answer generation
│   ├── memory/             # Shared memory buffer
│   └── tools/              # Tool implementations
│       ├── google_search/  # Serper.dev wrapper
│       ├── wikipedia_search/
│       ├── python_coder/
│       └── sql_executor/   # Spider SQLite executor
├── benchmarks/             # Benchmark data
│   ├── bamboogle/
│   ├── hotpotqa/
│   ├── musique/
│   ├── twowiki/
│   ├── gaia/
│   └── text2sql/           # Spider dataset
├── modal_deploy/           # Modal GPU scripts
│   ├── serve_planner.py    # 7B inference server
│   ├── serve_grpo_checkpoint.py  # Trained model server
│   └── train_flow_grpo.py  # Flow-GRPO training
├── scripts/
│   ├── run_benchmark.py    # Main benchmark runner
│   ├── evaluate.py         # LLM-as-judge scoring
│   └── show_results.py     # Results summary table
└── results/                # Output files (gitignored)
```

---

## Key Implementation Notes

1. **Answer extraction**: All benchmark queries include `"output the final answer enclosed in <answer> and </answer> tags"` — critical for LLM-as-judge to find the answer.

2. **Google Search**: Uses Serper.dev (POST to `https://google.serper.dev/search`) instead of Google Custom Search JSON API.

3. **Modal deployment**: Use `scaledown_window` (not `container_idle_timeout`), GPU as string `"A10G"` (not `modal.gpu.A10G()`). Pre-bake model weights via `.run_commands()` to avoid cold-start timeouts.

4. **Flow-GRPO**: Binary reward broadcast to all planner turns in a trajectory. Group normalization across G=8 trajectories. PPO-clipped objective + KL penalty against frozen reference.

---

## Citation

```bibtex
@article{huang2025agentflow,
  title={AgentFlow: Benchmarking Agentic Workflow Generation},
  author={Huang, Qin and others},
  year={2025}
}
```
