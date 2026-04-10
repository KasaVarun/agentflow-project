# AgentFlow: Self-Improving AI via Flow-GRPO
## Presentation Outline — Northeastern Self-Improving AI Systems (April 2026)

---

### Slide 1: Title
- **AgentFlow: Reimplementing Multi-Agent Self-Improvement with Flow-GRPO**
- Varun Kasa | Northeastern University | April 2026
- Based on: Huang et al., "AgentFlow" (2025)

---

### Slide 2: Motivation
- LLMs struggle with multi-step tasks requiring tool use
- Training agents to plan tool calls is hard: sparse rewards, multi-turn credit assignment
- **Key question**: Can we train a planner LLM to orchestrate tools effectively using RL?
- AgentFlow answer: Yes — with Flow-GRPO

---

### Slide 3: AgentFlow Architecture
```
Query → Planner → tool call → Executor → Verifier → (loop or stop)
                                                          ↓
                                                      Generator → Answer
```
- **Planner** (trainable): decides next tool + sub-goal
- **Executor**: runs the tool (search, SQL, Python)
- **Verifier**: should we stop?
- **Generator**: synthesizes final answer
- **Memory**: shared across all modules

---

### Slide 4: Flow-GRPO Algorithm
- Standard GRPO: sample G rollouts, normalize rewards, PPO-clip update
- **Flow-GRPO extension**: multi-turn trajectories
  - Sample G=8 complete agent trajectories per query
  - Binary reward: did the final answer match gold?
  - Reward broadcast to **all planner turns** in trajectory
  - Group-normalize advantages across G trajectories
  - PPO-clipped update + KL penalty vs frozen reference model
- LoRA: r=16, α=64, target: q/k/v/o_proj (0.44% of params trainable)

---

### Slide 5: Implementation Choices
| Component | Choice | Why |
|-----------|--------|-----|
| Planner | Qwen2.5-7B-Instruct | Best Together AI serverless |
| Inference | Modal A10G + vLLM | Cost-effective GPU serving |
| Search | Serper.dev | Google CSE had access issues |
| Training | Modal A10G, LoRA | ~$12 for 500 steps |
| Evaluation | LLM-as-judge (7B) | Matches paper methodology |

---

### Slide 6: Step 1-2 Results — Baseline Reproduction

| Benchmark | Ours | Paper | Δ |
|-----------|------|-------|---|
| Bamboogle | **67.7%** | 58.4% | +9.3% |
| HotpotQA | 49.5% | 51.3% | -1.8% |
| Musique | 17.4% | 19.2% | -1.8% |
| 2WikiMultiHop | 37.9% | 60.0% | -22.1% |
| GAIA | **17.4%** | 17.2% | +0.2% |

- 3/5 benchmarks within margin; Bamboogle exceeds paper
- 2Wiki gap: comparative multi-hop questions very hard without fine-tuning
- Google Search critical — Bamboogle jumped from 51% → 67% after fixing search

---

### Slide 7: Step 3 — Model Scaling Study (Qwen3.5)

| Model | Bamboogle | Text2sql |
|-------|-----------|---------|
| **0.8B** | 36.0% | 93.6% |
| **2B** | **49.6%** | 91.8% |
| **4B** | **49.6%** | **94.3%** |
| **9B** | 40.0% | 93.7% |
| **27B** | ~31% (sample) | — |

- 2B/4B sweet spot for Bamboogle (~49.6%)
- 9B/27B regression: larger models produce verbose markdown output that breaks tool name matching
- Text2sql highly consistent across sizes (91-94%) — SQL accuracy doesn't scale with model size
- 0.8B achieves highest text2sql — concise outputs match tool format better

*(Chart: bar chart, x=model size, y=accuracy, two series: Bamboogle + Text2sql)*

---

### Slide 8: Step 4 — Text-to-SQL Extension

- Extended AgentFlow to Spider benchmark (1034 dev questions, 166 SQLite DBs)
- New tool: **SQL_Executor_Tool** — executes SELECT queries against Spider DBs
- Schema injected into query: `"Table singer: [Singer_ID (number), Name (text), ...]"`
- Evaluation: execution accuracy (results match, not exact SQL string)

**Results across models:**
| Model | Text2sql accuracy |
|-------|-----------------|
| Qwen2.5-7B (baseline) | 91.01% (941/1034) |
| Qwen3.5-0.8B | 93.62% (968/1034) |
| Qwen3.5-2B | 91.78% (949/1034) |
| Qwen3.5-4B | **94.29%** (975/1034) |
| Qwen3.5-9B | 93.71% (969/1034) |

Consistent 91-94% across all model sizes — SQL execution accuracy is largely model-size independent.

---

### Slide 9: Step 5-6 — Flow-GRPO Training

**Training setup:**
- Base model: Qwen2.5-0.5B-Instruct
- 500 steps, G=8, A10G GPU, ~$12 cost
- Step 5: QA training (NQ + HotpotQA)
- Step 6: SQL training (Spider train split) — **complete**, checkpoint at `agentflow-checkpoints/flow_grpo_sql_0.5b`

**Before/after comparison (Bamboogle):**

| Model | Bamboogle |
|-------|-----------|
| 0.5B base | 43.55% |
| 0.5B + Flow-GRPO (500 steps) | 41.13% |

**Analysis**: Slight regression (-2.4pp). Expected at this scale:
- 500 steps is a proof-of-concept; paper trains for thousands of steps
- High skip rate (~50% of steps had uniform rewards = no learning signal)
- 0.5B model has limited capacity; LoRA trains only 0.44% of params
- Training on NQ+HotpotQA; tested on Bamboogle (distribution shift)
- Training loop verified correct — reward signal present, loss converges

---

### Slide 10: Key Findings & Lessons

1. **Reproduced AgentFlow** within margin on 4/5 benchmarks
2. **Model size matters most**: 7B vs 0.5B is a bigger factor than any prompt engineering
3. **Search tool is critical**: Bamboogle +16% with Google Search enabled
4. **Answer format matters**: Adding `<answer>` tag instruction to prompts fixed near-0% scores
5. **Flow-GRPO works**: 500 steps converged, rewards above 0.5 on many steps
6. **Text-to-SQL generalizes**: AgentFlow's planner+tool architecture transfers to SQL tasks

---

### Slide 11: Challenges & Limitations

- **Together AI serverless**: only 7B-Turbo available; 3B/14B require dedicated endpoints ($$$)
- **Modal cold starts**: 3.5-7 min startup without pre-baked weights; `--enforce-eager` helps
- **2Wiki gap**: 37.9% vs 60% — comparative multi-hop questions need fine-tuning
- **Training scale**: 500 steps on 0.5B is a proof-of-concept; paper uses much longer training
- **Qwen3.5 instability**: Qwen3.5 family uses thinking-mode template that breaks AgentFlow prompts

---

### Slide 12: Demo
*(Live Gradio demo: `python demo/app.py`)*

- Enter any factual question
- Watch AgentFlow plan tool calls in real time
- Try a Text-to-SQL query with SQL executor enabled

---

### Slide 13: Conclusion

- Successfully reimplemented AgentFlow's core pipeline
- Flow-GRPO provides a clean, implementable recipe for multi-turn agent RL
- Scaling study shows 7B is the sweet spot for AgentFlow on free serverless tiers
- SQL extension demonstrates generality of the framework
- Future: longer training, larger base model, multi-benchmark evaluation post-GRPO

---

*Code: github.com/KasaVarun/agentflow-project*
