# MLOps-Bench: A Procedural Benchmark for Evaluating AI Agents on ML Pipeline Debugging

## Paper Outline

**Target venue:** NeurIPS 2026 Datasets & Benchmarks Track (submission deadline ~June 2026)
**Alternative venues:** ICML 2026, EMNLP 2026, or arXiv preprint for priority

---

### Abstract (~250 words)

- Problem: Production ML pipelines fail in ways that require systematic investigation — config errors, data leakage, silent evaluation bugs. No existing benchmark evaluates AI agents on this task.
- Gap: SWE-bench evaluates general code repair; DebugBench evaluates syntax/logic debugging. Neither covers ML-specific failures involving training logs, preprocessing pipelines, evaluation metrics, and cross-artifact reasoning.
- Contribution: MLOps-Bench, a procedurally-generated benchmark where agents investigate broken training runs by reading artifacts (configs, logs, preprocessing code, eval results) and submitting structured diagnoses. 9 bug types across 3 difficulty tiers, deterministic grading, dense rewards.
- Results: Frontier models score ~0.42/0.28/0.15 (easy/medium/hard) without task-specific prompting. Silent evaluation bugs (hard tier) — which produce no error signals — remain genuinely challenging.
- Availability: Open-source, OpenEnv-compatible, Docker-deployable.

---

### 1. Introduction (~1.5 pages)

**Opening hook:** "A training job finishes overnight. The loss chart shows NaN. Or worse — everything looks fine, but test accuracy is 30% below validation with no error thrown."

**The problem:**
- ML pipeline debugging is the #1 time sink in production MLOps
- Requires cross-artifact reasoning: correlating logs with configs with preprocessing code
- Different from general code debugging — involves statistical reasoning, data distribution analysis, pipeline version tracking
- Three distinct failure modes with different signal characteristics:
  - Loud failures (config errors) — obvious in logs
  - Subtle failures (data leakage) — visible only by correlating multiple artifacts
  - Silent failures (eval bugs) — no error signal at all, only metric anomalies

**The gap:**
- SWE-bench (Jimenez et al., 2024): general GitHub issues, binary pass/fail
- DebugBench (Tian et al., 2024): code syntax/logic bugs, not ML-specific
- DeepDiagnosis (Wardat et al., 2021): automated tool, not agent benchmark
- No benchmark exists for evaluating AI agents on ML pipeline debugging

**Our contribution:**
1. MLOps-Bench: first benchmark for ML pipeline debugging by AI agents
2. Procedural artifact generation: unlimited episodes via seed, internally consistent artifacts with planted faults
3. Dense 4-component grading: category + file + field + fix scoring with partial credit
4. Asymmetric penalty design modeling real-world incident severity
5. Empirical evaluation showing frontier models struggle on silent bugs

---

### 2. Related Work (~1.5 pages)

**2.1 Code Debugging Benchmarks**
- DebugBench (Tian et al., ACL 2024): 4,253 instances, GPT-4 planted bugs in LeetCode code
- MdEval (Liu et al., 2024): multilingual code debugging, 3,600 samples
- RepairBench (Silva et al., 2024): leaderboard for frontier model program repair
- FixEval (Haque et al., 2022): execution-based evaluation of code fixes
- **Gap:** All focus on standalone code bugs (syntax, logic, algorithms). None involve ML-specific artifacts.

**2.2 Software Engineering Agent Benchmarks**
- SWE-bench (Jimenez et al., ICLR 2024): 2,294 real GitHub issues, agent must edit code to pass tests
- SWE-Gym (Pan et al., 2024): training environment variant of SWE-bench
- R2E-Gym (Jain et al., 2025): procedurally curated executable gym
- InterCode (Yang et al., 2023): interactive coding with execution feedback
- **Gap:** General SE tasks. No ML domain specialization, no training log analysis, no cross-artifact diagnosis.

**2.3 ML System Debugging Tools**
- DeepDiagnosis (Wardat et al., 2021): automated DL fault diagnosis
- DeepLocalize (Wardat et al., 2021): fault localization for DNNs
- DeepFD (Cao et al., 2022): automated fault diagnosis and localization
- Cockpit (Schneider et al., 2021): real-time training diagnostics
- DREAM (Zhang et al., 2024): debugging AutoML pipelines
- **Gap:** These are tools, not agent evaluation benchmarks. They detect bugs; they don't evaluate whether agents can diagnose bugs.

**2.4 Bug Databases and Procedural Generation**
- Defects4J: 835 real Java bugs (the gold standard for APR)
- BugsInPy: real Python bugs
- HyperPUT (Felici et al., 2022): procedurally generated synthetic faulty programs
- **Connection:** Our approach is most similar to HyperPUT in philosophy (procedural fault generation) but applied to ML pipelines, not general programs.

**2.5 ML Failure Taxonomies**
- Ma et al. (2025): comprehensive study of bugs in distributed DL systems
- Jiang et al. (2025): bugs in LLM libraries
- Lai et al. (2022): comparative analysis of bugs in open-source ML projects
- **Connection:** These empirical studies inform our bug catalogue design. Our 9 bug types are grounded in the most common failure modes identified by these taxonomies.

---

### 3. MLOps-Bench Environment (~3 pages)

**3.1 Problem Formulation**
- POMDP formulation: agent observes partial state (artifacts read so far), takes structured actions, receives dense rewards
- State space: 6 artifacts × content, investigation history, step count
- Action space: 8 discrete structured actions (read_config, read_logs, check_dataset_stats, inspect_preprocessing, read_eval_results, run_sanity_check, query_artifact, submit_diagnosis)
- Observation space: task description, run summary, artifact metadata, investigation progress, last action result
- Episode: reset → investigate → diagnose → grade

**3.2 Procedural Artifact Generation**
- Seed-based determinism: Random(seed) for bug selection, artifact variation
- 6 artifact types: config.yaml, train.log, dataset_stats.json, preprocessing.py, eval_results.json, model_card.json
- Internal consistency: config matches logs, stats match preprocessing, eval matches training
- Planted fault: exactly one bug from the task's 3-bug pool
- Variety: 5 model configs, 4 optimizers, 4 schedulers, variable epochs/batch sizes

**3.3 Bug Catalogue**
- 9 bugs across 3 difficulty tiers
- Easy (config errors): exploding_lr, wrong_optimizer, batch_size_overflow
- Medium (data leakage): scaler_leak, overlap_splits, wrong_split_ratio
- Hard (silent eval bugs): label_encoder_mismatch, metric_swap, tokenizer_drift
- Each bug defined by: category, file, field, gold_fix, difficulty
- Bug design grounded in empirical ML failure taxonomies (cite Ma et al., Lai et al.)

**3.4 Sanity Check Engine**
- 8 computed diagnostic checks (not just artifact reads)
- Each check returns structured results grounded in the generated artifacts
- Checks detect specific failure patterns (gradient explosion, data leakage, metric gaps, encoder mismatches)
- Models the diagnostic tools a senior ML engineer would use

**3.5 Reward Design**
- Dense per-step rewards: +0.02 new artifact, -0.02 duplicate, +0.01 new sanity check
- Terminal 4-component grading: category (0.15) + file (0.25) + field (0.30) + fix (0.30)
- Keyword/substring matching: deterministic, reproducible, no LLM judge
- Hard task asymmetric penalty: 1.5x on missed components when score < 0.70
- Design rationale: partial credit enables continuous score spectrum, not just pass/fail

---

### 4. Experiments (~2 pages)

**4.1 Experimental Setup**
- Models evaluated: [TODO: GPT-4o, Claude 3.5 Sonnet, Gemini 2.5 Flash, Llama 3.1 70B, Qwen2.5-72B]
- Seeds: [TODO: 10 seeds per task, report mean ± std]
- Prompting: system prompt with investigation strategy, no task-specific hints
- Ablations: with/without BUG_REFERENCE at diagnosis time

**4.2 Main Results**

Table: Mean score ± std across 10 seeds

| Model | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| GPT-4o | TODO | TODO | TODO | TODO |
| Claude 3.5 Sonnet | TODO | TODO | TODO | TODO |
| Gemini 2.5 Flash | TODO | TODO | TODO | TODO |
| Llama 3.1 70B | TODO | TODO | TODO | TODO |
| Qwen2.5-72B | ~0.42 | ~0.28 | ~0.15 | ~0.28 |

**4.3 Difficulty Validation**
- Show score distributions per tier confirm difficulty ordering
- Hard tasks genuinely challenge frontier models (silent bugs with no error signal)

**4.4 Ablation Studies**
- Dense vs sparse rewards: does per-step reward signal improve investigation quality?
- With/without sanity checks: do computed diagnostics help diagnosis accuracy?
- Step budget: how does max_steps affect score?
- Artifact read order: does investigation strategy matter?

**4.5 Error Analysis**
- What bugs do models get wrong and why?
- Which diagnosis components are hardest (category vs file vs field vs fix)?
- Do models investigate thoroughly or submit prematurely?
- Hard task analysis: which silent bugs are most challenging?

---

### 5. Analysis & Discussion (~1.5 pages)

**5.1 What Makes ML Debugging Hard for Agents?**
- Cross-artifact reasoning: must correlate signals across logs, code, stats
- Absence reasoning: silent bugs produce no positive error signal
- Domain knowledge: requires understanding of StandardScaler semantics, LabelEncoder behavior, tokenizer versioning

**5.2 Comparison with Existing Benchmarks**
- SWE-bench: binary pass/fail vs our continuous 4-component scoring
- DebugBench: single-file code bugs vs multi-artifact ML pipeline bugs
- Complementary, not competing: different skills tested

**5.3 Limitations**
- 9 bug types is limited (but extensible)
- Synthetic artifacts (but procedurally realistic)
- No multi-bug episodes (one fault per episode)
- No runtime/deployment bugs (training-time only)
- English-only artifacts

**5.4 Broader Impact**
- Potential to improve automated MLOps tooling
- Training signal for ML debugging agents
- Educational tool for junior ML engineers

---

### 6. Conclusion (~0.5 pages)

- First benchmark for AI agent evaluation on ML pipeline debugging
- Fills clear gap between code debugging (DebugBench) and SE benchmarks (SWE-bench)
- Procedural generation enables unlimited, reproducible evaluation
- Silent evaluation bugs remain genuinely challenging for frontier models
- Open-source, Docker-deployable, OpenEnv-compatible
- Future work: more bug types, multi-bug episodes, deployment-stage failures, human baseline comparison

---

### Appendix

- A: Full bug catalogue with gold fixes
- B: Example generated artifacts for each bug type
- C: Sanity check specifications
- D: Reward function formal definition
- E: Hyperparameter sensitivity analysis

---

## Experiment Plan (what you need to run)

### Priority 1: Multi-model evaluation (required for paper)
- Run inference.py against 4-5 models across 10 seeds each
- Models: GPT-4o, Claude Sonnet, Gemini Flash, Llama 70B, Qwen 72B
- This produces the main results table

### Priority 2: Ablation — with/without BUG_REFERENCE
- Run each model with and without the diagnosis reference
- Shows genuine reasoning ability vs prompted lookup

### Priority 3: Error analysis
- Categorize failures by component (category/file/field/fix)
- Identify which bugs are hardest for which models

### Priority 4: Human baseline (nice to have)
- Get 3-5 ML engineers to solve 5 episodes each
- Informal but adds strong credibility

### Priority 5: Investigation strategy analysis
- Track artifact read order across models
- Compare with optimal strategy defined in task descriptions
