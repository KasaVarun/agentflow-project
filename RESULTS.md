# Results

Measured results from this reimplementation. All numbers observed on the
configurations described in the README.

## Baseline reproduction: Qwen2.5-7B-Instruct

Reimplemented four-module pipeline (Planner, Executor, Verifier, Generator over
a shared memory buffer) evaluated against the published targets.

| Benchmark | This reimplementation | Published target | Delta |
|---|---|---|---|
| Bamboogle | **67.74%** | 58.4% | **+9.34** |
| GAIA | **17.36%** | 17.2% | **+0.16** |
| HotpotQA | 49.48% | 51.3% | −1.82 |
| Musique | 17.44% | 19.2% | −1.76 |
| 2WikiMultiHop | 37.89% | 60.0% | −22.11 |

Two benchmarks exceeded the published baseline; three fell short. The
2WikiMultiHop gap is the largest and is discussed below.

## Model-size sweep: Qwen3.5 on Bamboogle

Tooling, prompts, and judging protocol held constant across all five sizes.

| Model size | Bamboogle accuracy |
|---|---|
| 0.8B | 36.0% |
| 2B | **49.6%** |
| 4B | **49.6%** |
| 9B | 40.0% |
| 27B | 31.0% |

**Accuracy is non-monotonic in model size.** Performance peaks in the 2B–4B
range and then degrades, with the 27B model scoring *below* the 0.8B. Larger
models are not uniformly better at this agentic multi-hop task under a fixed
tool and prompt configuration.

This is the most interesting finding in the project. Plausible mechanisms worth
testing: larger models over-hedging and refusing to commit to a tool call,
drift in instruction-following against the fixed planner prompt, or the prompt
being implicitly tuned to smaller-model behaviour. Distinguishing between these
would require per-trajectory failure analysis, which this run did not do.

## Text-to-SQL: Spider dev execution accuracy

| Model | Execution accuracy |
|---|---|
| Qwen2.5-7B-Instruct (baseline) | 91.01% |
| Qwen3.5-0.8B | 93.62% |
| Qwen3.5-2B | 91.78% |
| Qwen3.5-4B | **94.29%** |
| Qwen3.5-9B | 93.71% |

**Every Qwen3.5 size beat the 7B baseline**, including the 0.8B. Unlike
multi-hop QA, Text-to-SQL performance is close to flat with respect to model
scale here — the 0.8B is within 0.7 points of the 4B best. For this task the
bottleneck appears to be schema grounding and the execution loop rather than
planner capacity.

## Flow-GRPO planner training

Only the Planner module is trained; Executor, Verifier, and Generator stay
frozen.

| Parameter | Value |
|---|---|
| Trained model | Qwen3.5-0.8B-Instruct |
| Steps | 50 |
| LoRA config | r=16, alpha=64, targets `q_proj` `k_proj` `v_proj` `o_proj` |
| Group size (G) | 8 |
| PPO clip epsilon | 0.2 |
| KL coefficient | 0.01 |

Two separate runs:

- **QA run** — Natural Questions + HotpotQA
- **SQL run** — Spider

## Infrastructure

| Component | Choice |
|---|---|
| Training | Modal A10G |
| 7B baseline inference | Together AI serverless |
| Trained checkpoint serving | vLLM on Modal A10G |
| Search tool | Serper.dev Google Search API |
| Other tools | Wikipedia, Python coder, SQL executor |

## Known gaps

- **2WikiMultiHop underperforms badly (37.89% vs 60.0%).** This benchmark leans
  on comparative-fact questions that need multi-hop entity lookup, and the
  retrieval path here is a search-API wrapper rather than an entity-aware
  retriever. This is the clearest target for improvement.
- **No post-training evaluation** of the Flow-GRPO checkpoints against the
  pre-training planner baseline. The training runs completed and checkpoints
  were saved, but the before/after comparison on Bamboogle and Spider has not
  been run. This is the single most valuable missing number in the project.
- Retrieval uses search snippets only. There is no embedding index, chunking
  strategy, hybrid retrieval, or reranking stage.
